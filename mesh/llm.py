# SPDX-License-Identifier: Apache-2.0
"""
LLM client for the mesh.

Multi-backend async client supporting:
- OpenAI-compatible APIs (openai, local vLLM, etc.)
- Anthropic Claude (native API)
- Claude Code subprocess (via `claude -p`)
- Z.AI (via Claude Code with Z.AI proxy)
- Codex CLI subprocess
- Mesh harness (TAOR loop)

Uses XML-wrapped conversation format for multi-party conversations.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import logging
import pwd
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolRegistry, ToolCall


# =============================================================================
# Streaming Callbacks for CC Tool Visibility
# =============================================================================

@dataclass
class CCToolEvent:
    """Represents a Claude Code tool call or result event."""
    event_type: Literal["tool_call", "tool_result"]
    call_id: str
    tool_name: str  # e.g., "cc:Read", "cc:Edit", "cc:Bash"
    # For tool_call: the input arguments
    # For tool_result: the output content
    data: dict | str


class LLMStreamCallback(Protocol):
    """Protocol for streaming callbacks during LLM completion."""

    def on_cc_tool_event(self, event: CCToolEvent) -> None:
        """Called when a CC tool call or result is observed."""
        ...

    def on_todos(self, todos: list[dict]) -> None:
        """Called when TodoWrite updates are observed."""
        ...

import httpx
import random
import re

logger = logging.getLogger(__name__)

# =============================================================================
# Prompt Sanitization
# =============================================================================

# DeepSeek internal token markers: <｜｜NAMESPACE｜｜tagname ...> where ｜ is U+FF5C.
# Structure: <  optional-/  ｜｜  word  ｜｜  tagname+attrs  >
# These leak into agent output and cause 400 errors when re-sent as input.
_FULLWIDTH_PIPE_TAG_RE = re.compile(r'</?｜｜\w+｜｜[^>]*>')


def sanitize_prompt(text: str) -> str:
    """Strip model-internal token markers that cause API 400 errors.

    Removes DeepSeek DSML tags (e.g. <｜｜DSML｜｜tool_calls>) and similar
    fullwidth-pipe-delimited control sequences. These are internal to
    DeepSeek's tokenizer and must not appear in API request payloads.
    """
    cleaned = _FULLWIDTH_PIPE_TAG_RE.sub('', text)
    if cleaned != text:
        n_removed = len(_FULLWIDTH_PIPE_TAG_RE.findall(text))
        logger.debug("sanitize_prompt: stripped %d model-internal token markers", n_removed)
    return cleaned


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Sanitize message content in an OpenAI-format message list."""
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str) and _FULLWIDTH_PIPE_TAG_RE.search(content):
            msg = {**msg, "content": sanitize_prompt(content)}
        elif isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if _FULLWIDTH_PIPE_TAG_RE.search(text):
                        part = {**part, "text": sanitize_prompt(text)}
                new_parts.append(part)
            msg = {**msg, "content": new_parts}
        out.append(msg)
    return out


# =============================================================================
# Retry Configuration
# =============================================================================

# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Maximum number of retry attempts
MAX_RETRIES = 5

# Base delay for exponential backoff (seconds)
BASE_RETRY_DELAY = 2.0

# Maximum delay between retries (seconds)
MAX_RETRY_DELAY = 60.0


class MalformedResponseError(Exception):
    """Raised when an LLM response is missing expected fields (e.g., choices, message).

    This is treated as a transient error: providers occasionally return empty or
    truncated bodies under load, especially behind load balancers like OpenRouter.
    """
    pass


class MeshHarnessCrashError(Exception):
    """Raised when the mesh-harness subprocess exits with a non-zero returncode.

    Carries any partial assistant text that was emitted before the crash, so
    callers can choose to retry, fall back, or surface the partial output to the
    user with a clear failure indicator.

    Attributes:
        returncode: Subprocess exit code.
        partial_text: All assistant text blocks emitted before the crash, joined
            with double newlines. May be empty.
        stderr: Captured stderr output (truncated to first 2000 chars in str()).
    """

    def __init__(self, returncode: int, partial_text: str, stderr: str):
        self.returncode = returncode
        self.partial_text = partial_text
        self.stderr = stderr
        super().__init__(self._render())

    def _render(self) -> str:
        stderr_preview = self.stderr.strip()[:500] if self.stderr else "(no stderr)"
        return (
            f"Mesh harness subprocess exited {self.returncode}; "
            f"partial_text={len(self.partial_text)} chars; "
            f"stderr: {stderr_preview}"
        )

    def as_user_message(self) -> str:
        """Render as a user-facing string preserving partial output."""
        header = f"[HARNESS EXIT {self.returncode} — partial output follows]"
        if self.partial_text:
            return f"{header}\n\n{self.partial_text}"
        return f"{header}\n\n(no partial output captured)"


# Directory where mesh-harness crash records are persisted for diagnosis.
# Resolved via mesh.paths (real home from /etc/passwd), not $HOME — $HOME is
# a synthetic CC acct home when the process was launched from a CC session.
from .paths import resolve_path as _resolve_mesh_path
HARNESS_CRASH_LOG_DIR = _resolve_mesh_path("~/.mesh/harness-crashes")


def _persist_harness_crash_log(
    *,
    agent_label: str,
    returncode: int,
    stderr: str,
    partial_text: str,
    model: str,
    prompt_len: int,
    usage: dict,
    log_dir: str | None = None,
) -> str | None:
    """Persist a mesh-harness crash record to a per-agent log file.

    Returns the path written, or None if the persist failed (logged as a
    warning). Must never raise — crash-log persistence is strictly best-effort
    and must not prevent MeshHarnessCrashError from propagating.
    """
    target_dir = log_dir if log_dir is not None else HARNESS_CRASH_LOG_DIR
    try:
        os.makedirs(target_dir, exist_ok=True)
        # ISO 8601 with microseconds, colons replaced for filesystem safety.
        from datetime import datetime, timezone as _tz
        ts = datetime.now(_tz.utc).isoformat().replace(":", "-")
        safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in (agent_label or "unknown"))
        path = os.path.join(target_dir, f"{safe_label}-{ts}.log")
        record = {
            "timestamp": datetime.now(_tz.utc).isoformat(),
            "agent_label": agent_label or "unknown",
            "returncode": returncode,
            "model": model,
            "prompt_len": prompt_len,
            "usage": usage,
            "stderr": stderr,
            "partial_text": partial_text,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
        return path
    except Exception as e:
        logger.warning(f"Failed to persist harness crash log to {target_dir}: {e}")
        return None


def _harness_agent_label(config) -> str:
    """Best-effort agent identifier for a harness crash log filename.

    Fall-through order:
    1. Explicit agent_label (set by AgentNode at init)
    2. Basename of harness_agent_socket (encodes agent nickname when MCP is on)
    3. Basename of cc_binary
    4. "unknown"
    """
    try:
        label = getattr(config, "agent_label", "") or ""
        if label:
            return label
        sock = getattr(config, "harness_agent_socket", "") or ""
        if sock:
            base = os.path.basename(sock)
            for suffix in (".sock", ".socket"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
            if base:
                return base
        cc_bin = getattr(config, "cc_binary", "") or ""
        if cc_bin:
            return os.path.basename(cc_bin)
    except Exception:
        pass
    return "unknown"


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an exception is retryable."""
    # Malformed provider responses (missing choices/message fields)
    if isinstance(exc, MalformedResponseError):
        return True

    # HTTP status errors with retryable codes
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES

    # Connection errors are retryable
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return True

    # Read timeout is retryable
    if isinstance(exc, httpx.ReadTimeout):
        return True

    # Pool timeout is retryable
    if isinstance(exc, httpx.PoolTimeout):
        return True

    # Generic connection/timeout errors (covers Google, etc.)
    exc_name = type(exc).__name__.lower()
    exc_str = str(exc).lower()

    # Check for common retryable error patterns in exception name/message
    retryable_patterns = [
        "timeout", "timed out", "connection", "connect",
        "unavailable", "overloaded", "rate limit", "too many requests",
        "503", "502", "504", "429",
    ]
    for pattern in retryable_patterns:
        if pattern in exc_name or pattern in exc_str:
            return True

    return False


def _get_retry_delay(attempt: int) -> float:
    """Calculate delay for retry attempt using exponential backoff with jitter."""
    # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s...
    delay = BASE_RETRY_DELAY * (2 ** attempt)
    # Cap at max delay
    delay = min(delay, MAX_RETRY_DELAY)
    # Add jitter (±25%)
    jitter = delay * 0.25 * (random.random() * 2 - 1)
    return delay + jitter


def _get_retry_after(exc: Exception) -> float | None:
    """Extract retry-after hint (seconds) from a 429 response, if present.

    Honors both numeric ("30") and HTTP-date formats. Returns None if absent
    or unparseable.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    if exc.response.status_code != 429:
        return None
    header = exc.response.headers.get("retry-after")
    if not header:
        return None
    try:
        # Numeric seconds form
        return max(0.0, float(header.strip()))
    except (ValueError, TypeError):
        pass
    # HTTP-date form
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone
        dt = parsedate_to_datetime(header)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except Exception:
        return None


# Token counting
try:
    import tiktoken
    _encoder = tiktoken.encoding_for_model("gpt-4")
except ImportError:
    tiktoken = None
    _encoder = None


# =============================================================================
# Token Estimation
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text using tiktoken (GPT-4 encoding).

    Falls back to word-based approximation if tiktoken unavailable.
    """
    if _encoder is not None:
        return len(_encoder.encode(text, disallowed_special=()))
    # Fallback: rough word-based estimate (1.3 tokens per word)
    return int(len(text.split()) * 1.3)


def estimate_history_tokens(
    history: list["HistoryMessage"],
    base_overhead: int = 8500,
) -> int:
    """
    Estimate total tokens in a history list, including prompt overhead.

    Args:
        history: List of history messages
        base_overhead: Fixed overhead for system prompt (~1500) and tool
            definitions (~7000). Default 8500 covers typical assistant agents.

    Returns:
        Estimated token count for the full prompt.
    """
    total = base_overhead
    for msg in history:
        total += estimate_tokens(msg.content)
        # XML tags per message: from/to/timestamp tags add ~35 tokens
        total += 35
    return total


# =============================================================================
# Summarization Prompt
# =============================================================================

SUMMARIZATION_PROMPT = r"""You are extending an existing conversation summary with a new batch of recent turns.

Below you will see:
1. An existing summary (if any) covering earlier conversation
2. A batch of new turns to incorporate

Your job is to produce an UPDATED summary that merges the old summary with the new turns.

## Guidelines

- **Preserve** key information from the existing summary: decisions, file paths, pending tasks,
  user preferences, and architectural context that is still relevant.
- **Incorporate** the new turns with appropriate detail — capture requests, actions, outcomes,
  and any corrections or feedback.
- **Organize by topic** using section headers with timestamps and status:
  - `[COMPLETED]` — topic is finished, compress to 1-2 sentences
  - `[ACTIVE]` — topic is ongoing, preserve more detail
  - `[PENDING]` — task requested but not yet started
- **Compress stale topics**: completed topics from earlier sessions should be reduced to
  single-line summaries. The most recent active topic gets the most detail.
- **Preserve specifics**: file paths, PIDs, configuration values, exact commands,
  and error messages are important — don't drop them.

## Output Structure

Produce a structured summary with these sections:

1. **Topic Sections** (grouped by subject, with timestamps and status labels):
   Each topic should have: timestamp range, status, and a description of what happened.
   Most recent topics get the most detail; old completed topics get compressed.

2. **Pending Tasks**: Tasks explicitly requested but not yet completed.

3. **Current Work**: What was being worked on most recently, with specifics.

4. **Key Artifacts**: Important files, configs, or resources that are actively relevant.

CONVERSATION TO SUMMARIZE:

{span_text}

Provide your updated summary."""


_SUBPROCESS_ENV_ALLOWLIST: frozenset[str] = frozenset({
    # System
    "PATH", "HOME", "USER", "LOGNAME", "SHELL",
    "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "LC_COLLATE",
    "LC_NUMERIC", "LC_TIME", "LC_MONETARY",
    "TERM", "COLORTERM",
    "TMPDIR", "TMP", "TEMP",
    "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
    "SSH_AUTH_SOCK",
    # API keys
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
    "EXA_API_KEY", "ZAI_API_KEY", "OPENROUTER_API_KEY",
    "SYNTHETIC_API_KEY", "GOOGLE_API_KEY",
    # Mesh
    "MESH_AUTH_TOKEN", "MESH_NODE_ID", "MESH_SOCKET_PATH", "MESH_ATTACHMENT_SECRET",
    # Notes server
    "RN_API_TOKEN", "RN_SERVER_BASE",
    # Python
    "VIRTUAL_ENV", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
})

_SUBPROCESS_ENV_PREFIXES: tuple[str, ...] = ("CLAUDE_", "ANTHROPIC_")


def _build_subprocess_env() -> dict[str, str]:
    """Build a curated subprocess environment from the allowlist.

    Blocks dangerous vars (LD_PRELOAD, LD_LIBRARY_PATH, DYLD_INSERT_LIBRARIES, etc.)
    by only including vars that are explicitly allowed or match safe prefixes.
    """
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _SUBPROCESS_ENV_ALLOWLIST or key.startswith(_SUBPROCESS_ENV_PREFIXES):
            env[key] = value
    return env


BackendType = Literal["openai", "anthropic", "claude-code", "claude-interactive", "zai", "codex", "mesh-harness"]
ReasoningEffort = Literal["none", "low", "medium", "high"]


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""
    # Backend type
    backend: BackendType = "openai"

    # Common settings
    model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7

    # OpenAI-compatible settings
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"

# Claude Code / Z.AI settings
    # These use environment variables for auth
    cc_allowed_tools: list[str] = field(default_factory=lambda: ["Read", "Edit", "Bash"])
    cc_fallback_homes: list[str] = field(default_factory=list)  # Fallback HOME dirs for multi-account CC

    # Z.AI-specific (env override for Claude CLI)
    zai_api_key: str = ""

    # CC subprocess environment overrides (generalizes ZAI pattern)
    # Merged into CC subprocess env (e.g., ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY)
    cc_env: dict[str, str] | None = None

    # CC subprocess idle timeout in seconds.  If the subprocess produces no
    # stdout data for this long, it is killed.  Resets on every chunk of output
    # (tool_call, tool_result, text, etc.).  0 = disabled.
    cc_idle_timeout: int = 600  # 10 minutes

    # Explicit thinking mode override for CC backend.
    # None = auto (CC decides from model name); True/False = force on/off.
    # CC has no --thinking flag; thinking is triggered by "openai/" model name prefix.
    cc_thinking: bool | None = None

    # CC --effort flag.  Controls reasoning depth, tool call volume, and
    # response thoroughness.  Empty string = let CC use its default.
    cc_effort: str = "high"

    # Path to Claude Code binary.  Allows pinning to a known-good version.
    # Empty string = auto-detect via shutil.which("claude").
    cc_binary: str = ""

    # MCP integration: when True, CC workers discover mesh tools via MCP sidecar
    # instead of XML <mesh_call> syntax.  Requires CC >= 2.1.114.
    cc_use_mcp: bool = False
    # Restrict Claude Code to an explicitly supplied MCP tool set.  This mode
    # disables every built-in tool, rejects interactive permission prompts,
    # and never enables the global permission bypass.  It is intended for
    # batch jobs (for example standing-digest folds) whose MCP server enforces
    # a narrower filesystem capability than Claude Code's native tools can.
    cc_safe_mcp_only: bool = False
    cc_safe_mcp_tools: list[str] = field(default_factory=list)

    # Worker briefing: router generates condensed briefing for CC workers
    # instead of passing full conversation history.  Briefing goes into
    # --system-prompt (durable under CC compaction).
    cc_worker_briefing: bool = False

    # Codex CLI settings
    codex_binary: str = ""  # Path to codex binary; empty = shutil.which("codex")

    # Codex subprocess idle timeout in seconds.  Codex can spend long stretches
    # in internal turns (large reasoning budget, multi-second LLM calls) without
    # writing to stdout, so idle-timeout watchdogs are usually wrong.  0 = disabled.
    codex_subprocess_idle_timeout: int = 0

    # Extra CLI arguments appended to `codex exec` invocation.  Use to enable
    # sandbox modes (`--sandbox read-only`), disable tools (`--disable shell_tool`),
    # set output schemas, etc.  When this list contains `--sandbox`, the default
    # `--dangerously-bypass-approvals-and-sandbox` flag is omitted (mutually exclusive).
    codex_extra_args: list[str] = field(default_factory=list)

    # Mesh harness settings (python -m mesh.harness exec)
    harness_python: str = ""  # Python binary; empty = sys.executable
    harness_backend: str = "anthropic"  # Sub-backend for the harness LLM calls
    harness_base_url: str = ""  # API base URL for harness sub-backend
    harness_api_key: str = ""  # API key for harness sub-backend
    harness_toolset: str = "legacy"  # "harness" (4-tool codex-style) or "legacy" (full mesh tools)
    harness_tools: str = ""  # Comma-separated tool names; overrides harness_toolset
    harness_system_prompt_file: str = ""  # System prompt file path
    harness_agent_socket: str = ""  # Unix socket path for routing agent-local tools to parent
    harness_soft_limit: int = 0  # Token soft limit for harness context (0 = use harness default)
    harness_controller_mode: str = "standard"  # "standard" | "plan_and_execute"
    harness_compaction_threshold_fraction: float = 0.40
    harness_max_phases: int = 15
    harness_assessor_backend: str = ""
    harness_assessor_model: str = ""
    harness_assessor_base_url: str = ""
    harness_assessor_api_key: str = ""
    harness_assessor_effort: str = ""
    harness_codex_assessor: bool = False
    harness_codex_assessor_binary: str = ""
    harness_codex_assessor_model: str = "o3"
    harness_codex_assessor_effort: str = "high"
    agent_label: str = ""  # Agent nickname for crash log attribution
    # Agent node ID — set this on any LLMClient that may spawn CC/codex/harness
    # subprocesses so MESH_NODE_ID is injected into their environment.
    node_id: str = ""

    # Anthropic API settings
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_thinking_budget: int | None = None  # budget_tokens for extended thinking

    # Prompt caching for Anthropic-compatible endpoints (Anthropic direct, z.ai, etc).
    # When True, places cache_control markers on tools and on per-message blocks within
    # the user content to enable rolling-prefix cache hits across auto-tool loop turns.
    anthropic_cache_enabled: bool = True

    # Thinking/Reasoning model settings
    # For OpenAI Responses API (o3, o4-mini, gpt-5, etc.):
    #   reasoning_effort: "none" | "low" | "medium" | "high"
    reasoning_effort: ReasoningEffort | None = None
    thinking_level: ReasoningEffort | None = None
    thinking_budget: int | None = None
    include_thoughts: bool = False

    # Auto-detect reasoning models based on model name patterns
    auto_detect_reasoning: bool = True

    # Cookie-based auth (e.g., "tamu" for TAMU Cloudflare Access)
    cookie_source: str = ""

    def is_reasoning_model(self, model: str | None = None) -> bool:
        """Check if the model is a reasoning/thinking model based on name patterns."""
        model = model or self.model
        model_lower = model.lower()

        # OpenAI reasoning models (includes Synthetic-hosted models using openai backend)
        openai_reasoning_patterns = [
            "o1", "o3", "o4",  # o-series reasoning models
            "gpt-5",  # GPT-5 has reasoning capabilities
            "kimi-k2-thinking", "kimi-k2.5",  # Moonshot Kimi models with thinking
            "deepseek-v3.2",  # DeepSeek V3.2 has native thinking
        ]

        if self.backend == "openai":
            return any(p in model_lower for p in openai_reasoning_patterns)

        return False

    def should_use_responses_api(self, model: str | None = None) -> bool:
        """Check if we should use OpenAI Responses API instead of Chat Completions."""
        if self.backend != "openai":
            return False

        # Proxied backends (e.g., Open WebUI via cookie_source) don't support
        # the Responses API — use Chat Completions with reasoning_effort instead.
        if self.cookie_source:
            return False

        model = model or self.model
        model_lower = model.lower()

        # Responses API is required for:
        # - o-series reasoning models (o1, o3, o4-mini, etc.)
        # - gpt-5 models (for reasoning capabilities)
        responses_api_models = ["o1", "o3", "o4", "gpt-5"]
        return any(p in model_lower for p in responses_api_models)

    def get_effective_reasoning_effort(self, model: str | None = None) -> str | None:
        """Get the effective reasoning effort for the current model.

        Returns the configured effort, or a default if auto-detect is enabled
        and we're using a reasoning model.
        """
        if self.reasoning_effort:
            return self.reasoning_effort

        if self.auto_detect_reasoning and self.is_reasoning_model(model):
            return "medium"  # Sensible default for reasoning models

        return None

    def get_effective_thinking_config(self, model: str | None = None) -> dict | None:
        """Get the effective thinking config for reasoning models.

        Returns a dict with thinking_level or thinking_budget based on model.
        """
        model = model or self.model
        model_lower = model.lower()

        # Explicit config takes precedence
        if self.thinking_level:
            return {"thinking_level": self.thinking_level}
        if self.thinking_budget is not None:
            return {"thinking_budget": self.thinking_budget}

        return None

    @classmethod
    def from_env(cls, backend: BackendType = "openai", prefix: str = "OPENAI") -> LLMConfig:
        """Load config from environment variables."""
        config = cls(backend=backend)

        if backend == "openai":
            config.api_key = os.environ.get(f"{prefix}_API_KEY", "")
            config.base_url = os.environ.get(f"{prefix}_BASE_URL", "https://api.openai.com/v1")
            config.model = os.environ.get(f"{prefix}_MODEL", "gpt-4")
        elif backend == "claude-code":
            config.model = os.environ.get("CLAUDE_MODEL", "sonnet")
        elif backend == "zai":
            config.zai_api_key = os.environ.get("ZAI_API_KEY", "")
            config.model = os.environ.get("ZAI_MODEL", "glm-4.7")

        return config


@dataclass
class ImageAttachment:
    """An image attached to a message."""
    data: str  # base64-encoded image data
    mime_type: str  # e.g., "image/jpeg"
    width: int | None = None
    height: int | None = None


@dataclass
class HistoryMessage:
    """A message in the conversation history."""
    from_node: str
    content: str
    timestamp: str
    to_node: str | None = None  # Destination (channel or user), if known
    images: list[ImageAttachment] | None = None  # Optional image attachments
    source: str = "persisted"  # "persisted" for saved history, "in_flight" for tool loop


def _strip_hallucinated_turns(text: str) -> str:
    """Strip hallucinated user turns from LLM output.

    Claude Code sometimes hallucinates follow-up conversation turns like:
    "Here's my response.\nHuman: thanks\n\nAssistant: You're welcome!"

    This strips everything starting from the first "Human:" on its own line.
    """
    import re
    # Match \nHuman: at start of line (common hallucination pattern)
    match = re.search(r'\n\s*Human:', text)
    if match:
        return text[:match.start()].rstrip()
    return text


class LLMClient:
    """
    Async LLM client for mesh agents.

    Supports multiple backends:
    - openai: OpenAI-compatible APIs (default)
    - anthropic: Anthropic Claude via native API
    - claude-code: Claude Code subprocess
    - claude-interactive: Claude Code interactive session via tmux wrapper
    - zai: Z.AI via Claude Code
    - codex: OpenAI Codex CLI subprocess
    - mesh-harness: Mesh harness subprocess (TAOR loop)

    Uses XML-wrapped conversation format where the entire history
    is sent as a single user message with XML structure.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._last_usage: dict | None = None  # Token usage from most recent LLM call
        self._cc_home_idx: int = 0  # Round-robin index for CC account load balancing
        self._cc_depleted: dict[int, float] = {}  # idx -> monotonic expiry time (skip until then)
        self.cc_effort: str | None = self.config.cc_effort or None
        self._ci_log_file: str | None = None  # Set by _complete_claude_interactive during CI worker execution

    async def __aenter__(self) -> LLMClient:
        if self.config.backend in ("openai", "anthropic"):
            self._client = httpx.AsyncClient(timeout=None)
        # claude-code, claude-interactive, zai, codex, mesh-harness don't need persistent clients
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure we have an HTTP client for OpenAI/Anthropic backend."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=None)
        return self._client

    @property
    def supports_native_reasoning_multiturn(self) -> bool:
        """Whether this backend needs native multi-turn with reasoning_content passback.

        DeepSeek v4-pro with thinking enabled requires that reasoning_content
        from tool-calling turns is passed back in subsequent API calls as part
        of native assistant messages.  Without this, the model restarts its
        reasoning chain from scratch on every tool-loop iteration.
        """
        if self.config.backend != "openai":
            return False
        if not self.config.base_url or "deepseek" not in self.config.base_url.lower():
            return False
        effort = self.config.get_effective_reasoning_effort(self.config.model)
        return bool(effort and effort != "none")

    def _log_usage(self) -> None:
        """Log token usage at INFO level if available."""
        if self._last_usage:
            u = self._last_usage
            parts = [f"in={u['input_tokens']} out={u['output_tokens']}"]
            if u.get("cache_creation_tokens") or u.get("cache_read_tokens"):
                parts.append(f"cache_create={u['cache_creation_tokens']} cache_read={u['cache_read_tokens']}")
            if u.get("reasoning_tokens"):
                parts.append(f"reasoning={u['reasoning_tokens']}")
            parts.append(f"total={u['total_tokens']}")
            logger.info(f"Token usage [{u['backend']}/{u['model']}]: {' '.join(parts)}")

    async def _retry_with_backoff(
        self,
        coro_factory: Callable[[], Any],
        operation_name: str = "LLM request",
    ) -> Any:
        """Execute an async operation with retry and exponential backoff.

        Args:
            coro_factory: A callable that returns a new coroutine for each attempt
            operation_name: Name of the operation for logging

        Returns:
            The result of the successful operation

        Raises:
            The last exception if all retries are exhausted
        """
        last_exception: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                return await coro_factory()
            except Exception as e:
                last_exception = e

                # Check if this error is retryable
                if not _is_retryable_error(e):
                    logger.debug(f"{operation_name} failed with non-retryable error: {e}")
                    raise

                # Check if we have retries left
                if attempt >= MAX_RETRIES:
                    logger.error(f"{operation_name} failed after {MAX_RETRIES + 1} attempts: {e}")
                    raise

                # Calculate delay: honor retry-after on 429 if present, else exponential backoff
                retry_after = _get_retry_after(e)
                if retry_after is not None:
                    delay = min(retry_after, MAX_RETRY_DELAY)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): "
                        f"HTTP 429 with retry-after={retry_after:.1f}s. Retrying in {delay:.1f}s..."
                    )
                elif isinstance(e, httpx.HTTPStatusError):
                    delay = _get_retry_delay(attempt)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): "
                        f"HTTP {e.response.status_code}. Retrying in {delay:.1f}s..."
                    )
                else:
                    delay = _get_retry_delay(attempt)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{MAX_RETRIES + 1}): "
                        f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                    )

                await asyncio.sleep(delay)

        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError(f"{operation_name} failed with unknown error")

    def format_history_xml(
        self,
        history: list[HistoryMessage],
        node_id: str,
        system_prompt: str = "",
        tool_prompt: str = "",
        instructions: str = "",
        trigger_msg: "Any | None" = None,
    ) -> str:
        """
        Format conversation history as XML for the LLM.

        Args:
            history: List of messages in chronological order
            node_id: This agent's ID (e.g., "agent:researcher")
            system_prompt: Custom system prompt for the agent
            tool_prompt: Generated tool usage instructions (from ToolRegistry)
            trigger_msg: If provided, the message that triggered this processing.
                         It is extracted from <history> and rendered as a separate
                         <message_received> block between history and instructions.

        Returns:
            Formatted prompt string with XML history
        """
        # Identify which history entry is the trigger (match last by from_node + content)
        trigger_idx = -1
        if trigger_msg is not None:
            t_from = getattr(trigger_msg, 'from_node', None)
            t_content = getattr(trigger_msg, 'content', None)
            if t_from and t_content:
                for i in range(len(history) - 1, -1, -1):
                    if history[i].from_node == t_from and history[i].content == t_content:
                        trigger_idx = i
                        break

        # Build history XML, skipping the trigger entry
        from .protocol import to_local_display
        history_xml = "<history>\n"
        for i, msg in enumerate(history):
            if i == trigger_idx:
                continue
            # Include to_node when available (e.g., for channel messages)
            ts = to_local_display(msg.timestamp)
            if msg.to_node:
                history_xml += f'<message from="{msg.from_node}" to="{msg.to_node}" timestamp="{ts}">\n'
            else:
                history_xml += f'<message from="{msg.from_node}" timestamp="{ts}">\n'
            history_xml += f"{msg.content}\n"
            history_xml += "</message>\n"
        history_xml += "</history>"

        # Build <message_received> block for the trigger
        message_received_xml = ""
        if trigger_idx >= 0:
            t = history[trigger_idx]
            ts = to_local_display(t.timestamp)
            attrs = f'from="{t.from_node}" timestamp="{ts}"'
            if t.to_node:
                attrs += f' to="{t.to_node}"'
            message_received_xml = f"<message_received {attrs}>\n{t.content}\n</message_received>"

        # Combine into full prompt
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"<system>\n{system_prompt}\n</system>")

        # Build identity section with parsed node info
        from .protocol import parse_node_id
        node_type, type_or_nick, nickname = parse_node_id(node_id)

        identity_lines = [f"You are {node_id}."]
        if node_type == "agent":
            identity_lines.append(f'Your agent type is "{type_or_nick}".')
            if nickname:
                identity_lines.append(f'Your nickname is "{nickname}" (how users will address you).')
        elif node_type == "user":
            identity_lines.append(f'Your nickname is "{type_or_nick}".')

        prompt_parts.append(f"<identity>\n" + "\n".join(identity_lines) + "\n</identity>")

        # Add tool prompt if provided
        if tool_prompt:
            prompt_parts.append(tool_prompt)

        prompt_parts.append(history_xml)

        # Insert <message_received> between history and instructions
        if message_received_xml:
            prompt_parts.append(message_received_xml)

        # Use provided instructions or fall back to generic prompt
        if instructions:
            prompt_parts.append(f"<instructions>\n{instructions}\n</instructions>")
        elif message_received_xml:
            prompt_parts.append("<instructions>\nRespond to the <message_received> above. The <history> is prior context.\n</instructions>")
        else:
            prompt_parts.append("<instructions>\nRespond to the most recent message in the conversation.\n</instructions>")

        return "\n\n".join(prompt_parts)

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        callback: LLMStreamCallback | None = None,
        mcp_config: str | None = None,
        cc_system_prompt: str | None = None,
        cc_watchdog: Any = None,
    ) -> str:
        """
        Send prompt to LLM and get response.

        Routes to appropriate backend based on config.
        For OpenAI, automatically uses Responses API for reasoning models (o3, o4, gpt-5).

        Args:
            prompt: The formatted prompt (XML-wrapped history)
            model: Override model (uses config default if None)
            max_tokens: Override max tokens
            temperature: Override temperature
            callback: Optional callback for streaming events (CC tool calls, etc.)

        Returns:
            The LLM's response text
        """
        model = model or self.config.model
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        backend = self.config.backend

        prompt = sanitize_prompt(prompt)

        if backend == "openai":
            # Use Responses API for reasoning models (o3, o4-mini, gpt-5, etc.)
            if self.config.should_use_responses_api(model):
                result = await self._complete_openai_responses(prompt, model, max_tokens)
            else:
                result = await self._complete_openai(prompt, model, max_tokens, temperature)
        elif backend == "anthropic":
            result = await self._complete_anthropic(prompt, model, max_tokens, temperature)
        elif backend == "claude-code":
            result = await self._complete_claude_code(prompt, model, callback=callback, mcp_config=mcp_config, cc_system_prompt=cc_system_prompt, cc_watchdog=cc_watchdog)
        elif backend == "zai":
            result = await self._complete_zai(prompt, model, callback=callback, mcp_config=mcp_config, cc_system_prompt=cc_system_prompt, cc_watchdog=cc_watchdog)
        elif backend == "codex":
            result = await self._complete_codex(prompt, model, callback=callback)
        elif backend == "claude-interactive":
            result = await self._complete_claude_interactive(prompt, model)
        elif backend == "mesh-harness":
            result = await self._complete_mesh_harness(prompt, model, callback=callback, system_prompt=cc_system_prompt)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._log_usage()
        return result

    async def _complete_openai(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
        images: list[ImageAttachment] | None = None,
    ) -> str | tuple[str, list[dict]]:
        """
        Complete using OpenAI-compatible API.

        Args:
            prompt: The prompt text
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            tools: Optional list of OpenAI-formatted tool definitions
            images: Optional list of images to include (for vision models)

        Returns:
            If tools is None or no tool calls: returns content string
            If tool calls present: returns (content, tool_calls) tuple
        """
        client = self._ensure_client()

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        # Only include Authorization header if api_key is set (supports local models)
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Inject cookie-based auth (e.g., TAMU Cloudflare Access)
        if self.config.cookie_source == "tamu":
            from .tamu_cookies import load_tamu_cookies
            cookies = load_tamu_cookies()
            if cookies:
                headers["Cookie"] = cookies
            else:
                raise RuntimeError(
                    "TAMU cookies expired or unavailable — "
                    "log into chat.tamu.ai in mesh-browser to refresh"
                )

        # Build message content - use multi-part format if images are present
        if images:
            # Multi-part content with images for vision models
            content_parts: list[dict] = [
                {"type": "text", "text": prompt}
            ]
            for img in images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.mime_type};base64,{img.data}",
                        "detail": "auto"  # Let the model decide resolution
                    }
                })
            user_content = content_parts
        else:
            user_content = prompt

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Open WebUI defaults to streaming — explicitly disable
        if self.config.cookie_source:
            payload["stream"] = False

        # Add tools if provided
        if tools:
            payload["tools"] = tools
            # Azure-hosted o3 rejects parallel_tool_calls; other models support it
            if not (self.config.cookie_source and "o3" in model.lower()):
                payload["parallel_tool_calls"] = True

        # Add reasoning_effort for Chat Completions reasoning models
        # This works with OpenAI o-series, and Synthetic's OpenAI-compatible endpoint
        reasoning_effort = self.config.get_effective_reasoning_effort(model)
        if reasoning_effort and reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort
            # DeepSeek v4-pro requires explicit thinking enablement alongside reasoning_effort.
            # Per docs: reasoning_effort selects effort level, but thinking must be enabled separately.
            if "deepseek" in (self.config.base_url or "").lower():
                payload["thinking"] = {"type": "enabled"}

        logger.debug(
            f"LLM request [openai]: model={model} url={url} prompt_len={len(prompt)} "
            f"max_tokens={max_tokens} temp={temperature} tools={len(tools) if tools else 0} "
            f"images={len(images) if images else 0} reasoning_effort={reasoning_effort}"
        )
        logger.debug(f"LLM prompt preview (first 500 chars): {prompt[:500]!r}...")

        async def _make_request() -> str | tuple[str, list[dict]]:
            response = await client.post(url, headers=headers, json=payload)

            # Handle models that don't support reasoning_effort (e.g., Qwen3-Coder on Synthetic)
            # Retry without reasoning_effort if the backend rejects it
            if response.status_code == 400 and "reasoning_effort" in payload:
                error_text = response.text
                effort_val = payload.get("reasoning_effort")
                # Backend that accepts reasoning_effort but not the specific value
                # (e.g., vLLM gpt-oss accepts only 'none|low|medium|high', not 'xhigh').
                # Downgrade rather than dropping entirely so reasoning stays enabled.
                if (
                    "reasoning_effort" in error_text
                    and effort_val == "xhigh"
                    and "high" in error_text
                ):
                    logger.info(
                        f"Model {model} rejects reasoning_effort='xhigh', "
                        f"downgrading to 'high' and retrying"
                    )
                    payload["reasoning_effort"] = "high"
                    response = await client.post(url, headers=headers, json=payload)
                elif "non-reasoning model" in error_text or "does not support reasoning" in error_text:
                    logger.info(f"Model {model} doesn't support reasoning_effort, retrying without it")
                    payload.pop("reasoning_effort")
                    response = await client.post(url, headers=headers, json=payload)

            # Always log 4xx response bodies — they carry the actionable error detail
            # and were previously dropped, making endpoint-shape bugs hard to diagnose.
            if 400 <= response.status_code < 500:
                logger.warning(
                    f"OpenAI-compatible {response.status_code} from {url} "
                    f"(model={model}): body[:1000]={response.text[:1000]!r}"
                )

            response.raise_for_status()

            try:
                data = response.json()
            except Exception as e:
                raise MalformedResponseError(
                    f"Failed to parse JSON response from {model}: {e}; body[:200]={response.text[:200]!r}"
                ) from e

            # Provider responses occasionally arrive without choices (load-balanced
            # backends like OpenRouter, transient upstream errors, truncated bodies).
            # Treat as transient and retry rather than crashing with KeyError.
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                err = data.get("error") or {}
                err_msg = err.get("message") if isinstance(err, dict) else str(err)
                raise MalformedResponseError(
                    f"OpenAI response missing 'choices' (model={model}): "
                    f"error={err_msg!r} body[:200]={str(data)[:200]!r}"
                )
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise MalformedResponseError(
                    f"OpenAI response choices[0] missing 'message' (model={model}): "
                    f"body[:200]={str(data)[:200]!r}"
                )
            content = message.get("content") or ""

            # Store raw message for native multi-turn reasoning passback (DeepSeek)
            self._last_raw_message = message

            # Capture reasoning_content from Chat Completions response
            # Models like GLM-4.7 via Synthetic return this when reasoning is enabled
            reasoning_content = message.get("reasoning_content") or ""
            self._last_reasoning_content = reasoning_content.strip() if reasoning_content else None

            # Some models (e.g. gpt-oss via vLLM) put their substantive response
            # entirely in reasoning_content with empty content — use it as fallback.
            if not content.strip() and reasoning_content.strip():
                content = reasoning_content.strip()

            usage = data.get("usage", {})
            ct_details = usage.get("completion_tokens_details") or {}
            # Cached prompt tokens: prefer the standard OpenAI mirror field
            # (`prompt_tokens_details.cached_tokens`); fall back to DeepSeek's
            # `prompt_cache_hit_tokens` when the standard field is absent.
            pt_details = usage.get("prompt_tokens_details") or {}
            cached_tokens = 0
            if isinstance(pt_details, dict):
                cached_tokens = pt_details.get("cached_tokens", 0) or 0
            if not cached_tokens:
                cached_tokens = usage.get("prompt_cache_hit_tokens", 0) or 0
            self._last_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_tokens": 0,
                "cache_read_tokens": cached_tokens,
                "reasoning_tokens": ct_details.get("reasoning_tokens", 0) if isinstance(ct_details, dict) else 0,
                "total_tokens": usage.get("total_tokens", 0) or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
                "backend": "openai",
                "model": model,
            }
            reasoning_tok = ct_details.get("reasoning_tokens", 0) if isinstance(ct_details, dict) else 0
            logger.debug(
                f"LLM response: len={len(content)} "
                f"prompt_tokens={usage.get('prompt_tokens', '?')} "
                f"completion_tokens={usage.get('completion_tokens', '?')} "
                f"reasoning_tokens={reasoning_tok}"
            )
            if reasoning_content:
                logger.debug(f"Reasoning content: {reasoning_content[:200]}...")
            logger.debug(f"LLM response content preview: {content[:500]!r}...")

            # Check for tool calls
            tool_calls = message.get("tool_calls")
            if tool_calls:
                logger.debug(f"LLM returned {len(tool_calls)} tool call(s)")
                return content, tool_calls

            return content

        return await self._retry_with_backoff(_make_request, f"OpenAI request ({model})")

    async def _complete_openai_responses(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        tools: list[dict] | None = None,
        images: list[ImageAttachment] | None = None,
    ) -> str | tuple[str, list[dict]]:
        """Complete using OpenAI Responses API (for reasoning models: o3, o4, gpt-5).

        The Responses API is the newer API that supports:
        - reasoning parameter with effort levels
        - reasoning_items in output for transparency
        - previous_response_id for multi-turn with preserved reasoning
        - native function calling via tools parameter
        - multimodal input with images (input_image type)

        Reference: https://platform.openai.com/docs/api-reference/responses

        Args:
            prompt: The prompt text
            model: Model name
            max_tokens: Maximum tokens to generate
            tools: Optional list of OpenAI-formatted tool definitions
            images: Optional list of images to include (for vision models)

        Returns:
            If tools is None or no tool calls: returns content string
            If tool calls present: returns (content, tool_calls) tuple
        """
        client = self._ensure_client()

        url = f"{self.config.base_url.rstrip('/')}/responses"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        # Only include Authorization header if api_key is set (supports local models)
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        # Inject cookie-based auth (e.g., TAMU Cloudflare Access)
        if self.config.cookie_source == "tamu":
            from .tamu_cookies import load_tamu_cookies
            cookies = load_tamu_cookies()
            if cookies:
                headers["Cookie"] = cookies
            else:
                raise RuntimeError(
                    "TAMU cookies expired or unavailable — "
                    "log into chat.tamu.ai in mesh-browser to refresh"
                )

        # Build the input - use message array format if images are present
        if images:
            # Multi-part input with images for vision models
            # Responses API uses messages with role + content containing input_text/input_image
            content_parts: list[dict] = [
                {"type": "input_text", "text": prompt}
            ]
            for img in images:
                content_parts.append({
                    "type": "input_image",
                    "image_url": f"data:{img.mime_type};base64,{img.data}",
                    "detail": "auto"  # Let the model decide resolution
                })
            # Wrap in a user message
            input_content: str | list[dict] = [
                {"role": "user", "content": content_parts}
            ]
        else:
            input_content = prompt

        # Build the payload
        payload: dict[str, Any] = {
            "model": model,
            "input": input_content,  # Responses API uses "input" (string or array)
            "max_output_tokens": max_tokens,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        # Add reasoning configuration
        reasoning_effort = self.config.get_effective_reasoning_effort(model)
        if reasoning_effort:
            reasoning_config: dict[str, Any] = {"effort": reasoning_effort}
            # Only request reasoning summaries if include_thoughts is enabled
            # (requires organization verification on OpenAI)
            if self.config.include_thoughts:
                reasoning_config["summary"] = "auto"
            payload["reasoning"] = reasoning_config

        logger.debug(
            f"LLM request [openai-responses]: model={model} url={url} prompt_len={len(prompt)} "
            f"max_tokens={max_tokens} reasoning_effort={reasoning_effort} tools={len(tools) if tools else 0} "
            f"images={len(images) if images else 0}"
        )
        logger.debug(f"LLM prompt preview (first 500 chars): {prompt[:500]!r}...")

        async def _make_request() -> str | tuple[str, list[dict]]:
            response = await client.post(url, headers=headers, json=payload)

            # Always log 4xx response bodies — they carry the actionable error detail.
            if 400 <= response.status_code < 500:
                logger.warning(
                    f"OpenAI Responses {response.status_code} from {url} "
                    f"(model={model}): body[:1000]={response.text[:1000]!r}"
                )

            response.raise_for_status()

            try:
                data = response.json()
            except Exception as e:
                raise MalformedResponseError(
                    f"Failed to parse JSON response from {model}: {e}; body[:200]={response.text[:200]!r}"
                ) from e

            # Extract the output text and tool calls
            # Responses API returns output items; we want text items and function_call items
            content = ""
            reasoning_content = ""
            tool_calls = []

            for item in data.get("output", []):
                item_type = item.get("type")
                if item_type == "message":
                    # Message items contain the actual response content
                    for content_item in item.get("content", []):
                        if content_item.get("type") == "output_text":
                            content += content_item.get("text", "")
                elif item_type == "reasoning":
                    # Reasoning items contain the model's thinking (if summary requested)
                    for summary in item.get("summary", []):
                        if summary.get("type") == "summary_text":
                            reasoning_content += summary.get("text", "") + "\n"
                elif item_type == "function_call":
                    # Function call items - convert to Chat Completions format for consistency
                    tool_calls.append({
                        "id": item.get("call_id", item.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        }
                    })

            # Store reasoning content for potential use
            self._last_reasoning_content = reasoning_content.strip() if reasoning_content else None

            # Store response ID for potential multi-turn continuation
            self._last_response_id = data.get("id")

            usage = data.get("usage", {})
            reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0) if isinstance(usage.get("output_tokens_details"), dict) else 0
            # Cached prompt tokens: Responses API uses `input_tokens_details.cached_tokens`.
            it_details = usage.get("input_tokens_details") or {}
            cached_tokens = 0
            if isinstance(it_details, dict):
                cached_tokens = it_details.get("cached_tokens", 0) or 0
            self._last_usage = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_creation_tokens": 0,
                "cache_read_tokens": cached_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "backend": "openai",
                "model": model,
            }
            logger.debug(
                f"LLM response: len={len(content)} "
                f"input_tokens={usage.get('input_tokens', '?')} "
                f"output_tokens={usage.get('output_tokens', '?')} "
                f"reasoning_tokens={reasoning_tokens}"
            )
            if reasoning_content:
                logger.debug(f"Reasoning summary: {reasoning_content[:200]}...")
            logger.debug(f"LLM response content preview: {content[:500]!r}...")

            # Return with tool calls if present
            if tool_calls:
                logger.debug(f"LLM returned {len(tool_calls)} tool call(s)")
                return content, tool_calls

            return content

        return await self._retry_with_backoff(_make_request, f"OpenAI Responses API ({model})")

    @staticmethod
    def _build_anthropic_cached_user_content(
        prompt: str,
        images: list[ImageAttachment] | None = None,
    ) -> list[dict]:
        """Build a multi-block user content array with cache_control markers for
        rolling prompt-prefix caching on Anthropic-compatible endpoints.

        Strategy: split the prompt at </message>\\n boundaries within <history>...
        </history>, producing one text block per history message plus framing
        blocks for the prefix (system/identity/tools) and suffix (closing tags,
        message_received, instructions). Place cache_control on the LAST history-
        message block so it acts as the rolling cache breakpoint.

        On the next request (one more message in history), the lookback walks
        back from the new last-message marker and finds the prior turn's hash
        at its previous position, giving cache_read for everything up to and
        including the previous turn's last message.

        Falls back to a single-block user content (still cache_control-marked)
        if the prompt doesn't have a recognizable <history>...</history> region
        or has fewer than 2 messages inside it.
        """
        # Detect <history>...</history> region
        h_open = prompt.find("<history>")
        h_close = prompt.find("</history>", h_open) if h_open >= 0 else -1

        # Default fallback: single text block with cache_control, plus optional images
        def _fallback() -> list[dict]:
            blocks: list[dict] = []
            if images:
                for img in images:
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.mime_type,
                            "data": img.data,
                        }
                    })
            blocks.append({
                "type": "text",
                "text": prompt,
                "cache_control": {"type": "ephemeral"},
            })
            return blocks

        if h_open < 0 or h_close < 0:
            return _fallback()

        # Split the history body on </message>\n. Note: </message_received>
        # ends with "_received>" so it won't match the literal "</message>\n".
        history_body = prompt[h_open:h_close]
        sep = "</message>\n"
        parts = history_body.split(sep)
        # Need at least 2 messages to benefit from per-message splitting.
        # parts shape with N messages: [<history>\n<message...>content_1, content_2, ..., content_N, ""]
        non_trailing = [p for p in parts[:-1]]  # drop trailing "" after final </message>\n
        if len(non_trailing) < 2:
            return _fallback()

        prefix_text = prompt[:h_open]  # everything before <history>
        suffix_text = prompt[h_close:]  # </history> ... onward (closing + msg_received + instructions)

        blocks: list[dict] = []

        # Image blocks first if present
        if images:
            for img in images:
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.mime_type,
                        "data": img.data,
                    }
                })

        # Block 0: prefix (system/identity/tools/header) + first message
        first_block_text = prefix_text + non_trailing[0] + sep
        blocks.append({"type": "text", "text": first_block_text})

        # Per-message blocks for messages 2..N
        for content in non_trailing[1:]:
            blocks.append({"type": "text", "text": content + sep})

        # Suffix block: </history> + <message_received> + <instructions>
        blocks.append({"type": "text", "text": suffix_text})

        # Mark cache_control on the LAST history-message block (second-to-last
        # block in the list). This is the rolling breakpoint that moves forward
        # one message per turn.
        if len(blocks) >= 2:
            blocks[-2]["cache_control"] = {"type": "ephemeral"}

        return blocks

    async def _complete_anthropic(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        tools: list[dict] | None = None,
        images: list[ImageAttachment] | None = None,
    ) -> str | tuple[str, list[dict]]:
        """Complete using Anthropic Messages API with optional extended thinking.

        The Anthropic Messages API supports:
        - Extended thinking via `thinking` parameter
        - Tool use with thinking blocks preserved across turns
        - Multi-modal input with images

        For Synthetic's Anthropic-compatible endpoint, use:
        - base_url: https://api.synthetic.new/anthropic/v1
        - Models: hf:zai-org/GLM-4.7, hf:deepseek-ai/DeepSeek-V3.2, etc.

        Reference: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking

        Args:
            prompt: The prompt text (will be wrapped as user message content)
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (ignored if thinking enabled)
            tools: Optional list of Anthropic-formatted tool definitions
            images: Optional list of images to include

        Returns:
            If tools is None or no tool calls: returns content string
            If tool calls present: returns (content, tool_calls) tuple
        """
        client = self._ensure_client()

        url = f"{self.config.anthropic_base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Build message content. When caching is enabled, split into per-message
        # blocks and place a rolling cache_control breakpoint on the last history
        # message — this lets call N+1 read-cache everything through the prior
        # turn. Otherwise fall back to the simple format.
        if self.config.anthropic_cache_enabled:
            user_content = self._build_anthropic_cached_user_content(prompt, images)
        elif images:
            content_parts: list[dict] = []
            for img in images:
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.mime_type,
                        "data": img.data,
                    }
                })
            content_parts.append({"type": "text", "text": prompt})
            user_content = content_parts
        else:
            user_content = prompt

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "max_tokens": max_tokens,
        }

        # Add temperature (only if thinking is not enabled, as thinking requires no temp modification)
        thinking_budget = self.config.anthropic_thinking_budget
        if thinking_budget is None:
            payload["temperature"] = temperature

        # Add extended thinking if configured
        if thinking_budget is not None and thinking_budget > 0:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        # Add tools if provided. When caching is enabled, mark the last tool
        # with cache_control so the system+tools prefix becomes a stable
        # cache breakpoint that survives across calls within the cache TTL.
        if tools:
            if self.config.anthropic_cache_enabled:
                tools_cached = [dict(t) for t in tools]
                tools_cached[-1] = {**tools_cached[-1], "cache_control": {"type": "ephemeral"}}
                payload["tools"] = tools_cached
            else:
                payload["tools"] = tools

        logger.debug(
            f"LLM request [anthropic]: model={model} url={url} prompt_len={len(prompt)} "
            f"max_tokens={max_tokens} temp={temperature if thinking_budget is None else 'N/A (thinking)'} "
            f"thinking_budget={thinking_budget} tools={len(tools) if tools else 0} "
            f"images={len(images) if images else 0}"
        )
        logger.debug(f"LLM prompt preview (first 500 chars): {prompt[:500]!r}...")

        async def _make_request() -> str | tuple[str, list[dict]]:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            try:
                data = response.json()
            except Exception as e:
                raise MalformedResponseError(
                    f"Failed to parse JSON response from {model}: {e}; body[:200]={response.text[:200]!r}"
                ) from e

            # Parse response content blocks
            content = ""
            thinking_content = ""
            tool_calls = []

            # Track thinking blocks for potential preservation in multi-turn
            self._last_thinking_blocks: list[dict] = []

            for block in data.get("content", []):
                block_type = block.get("type")

                if block_type == "thinking":
                    # Thinking block - preserve for multi-turn and optionally include in output
                    self._last_thinking_blocks.append(block)
                    if self.config.include_thoughts:
                        thinking_content += block.get("thinking", "") + "\n"

                elif block_type == "redacted_thinking":
                    # Redacted thinking - preserve for multi-turn (required for continuity)
                    self._last_thinking_blocks.append(block)

                elif block_type == "text":
                    content += block.get("text", "")

                elif block_type == "tool_use":
                    # Tool use block - convert to standard format
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        }
                    })

            # Store thinking content for potential use
            self._last_reasoning_content = thinking_content.strip() if thinking_content else None

            # Log usage
            usage = data.get("usage", {})
            self._last_usage = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                "reasoning_tokens": 0,
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "backend": "anthropic",
                "model": model,
            }
            logger.debug(
                f"LLM response: len={len(content)} "
                f"input_tokens={usage.get('input_tokens', '?')} "
                f"output_tokens={usage.get('output_tokens', '?')}"
            )
            if thinking_content:
                logger.debug(f"Thinking content preview: {thinking_content[:200]}...")
            logger.debug(f"LLM response content preview: {content[:500]!r}...")

            # Return with tool calls if present
            if tool_calls:
                logger.debug(f"LLM returned {len(tool_calls)} tool call(s)")
                return content, tool_calls

            return content

        return await self._retry_with_backoff(_make_request, f"Anthropic API ({model})")

    async def complete_multi_turn_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
    ) -> tuple[str, list["ToolCall"], dict]:
        """Anthropic Messages API multi-turn call with native message format.

        Counterpart to complete_multi_turn (OpenAI Chat Completions). Uses the
        Anthropic /messages endpoint with x-api-key auth, extended thinking,
        and prompt caching.

        Messages should be in Anthropic format:
        - system message extracted to top-level ``system`` parameter
        - assistant messages with content as list of blocks (text, tool_use,
          thinking, redacted_thinking)
        - user messages with tool_result content blocks

        Returns:
            (content, tool_calls, usage) — same contract as complete_multi_turn.
        """
        from .tools import ToolCall

        model = model or self.config.model
        max_tokens = self.config.max_tokens
        client = self._ensure_client()

        url = f"{self.config.anthropic_base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": self.config.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Separate system message from conversation messages.
        system_content: str | list[dict] | None = None
        conversation: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                conversation.append(msg)

        payload: dict[str, Any] = {
            "model": model,
            "messages": conversation,
            "max_tokens": max_tokens,
        }

        if system_content:
            if self.config.anthropic_cache_enabled and isinstance(system_content, str):
                payload["system"] = [
                    {"type": "text", "text": system_content, "cache_control": {"type": "ephemeral"}},
                ]
            else:
                payload["system"] = system_content

        thinking_budget = self.config.anthropic_thinking_budget
        if thinking_budget is None:
            payload["temperature"] = self.config.temperature

        if thinking_budget is not None and thinking_budget > 0:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        if tools:
            if self.config.anthropic_cache_enabled:
                tools_cached = [dict(t) for t in tools]
                tools_cached[-1] = {**tools_cached[-1], "cache_control": {"type": "ephemeral"}}
                payload["tools"] = tools_cached
            else:
                payload["tools"] = tools

        logger.debug(
            "LLM multi-turn-anthropic request: model=%s url=%s messages=%d tools=%d thinking=%s",
            model, url, len(conversation), len(tools) if tools else 0,
            thinking_budget or "off",
        )

        async def _make_request() -> tuple[str, list[ToolCall], dict]:
            response = await client.post(url, headers=headers, json=payload)

            if 400 <= response.status_code < 500:
                logger.warning(
                    "Multi-turn-anthropic %d from %s (model=%s): body[:1000]=%s",
                    response.status_code, url, model, response.text[:1000],
                )

            response.raise_for_status()

            try:
                data = response.json()
            except Exception as e:
                raise MalformedResponseError(
                    f"Failed to parse JSON response from {model}: {e}; body[:200]={response.text[:200]!r}"
                ) from e

            content = ""
            tool_calls: list[ToolCall] = []
            self._last_thinking_blocks = []

            for block in data.get("content", []):
                block_type = block.get("type")

                if block_type == "thinking":
                    self._last_thinking_blocks.append(block)
                    if self.config.include_thoughts:
                        content += block.get("thinking", "") + "\n"

                elif block_type == "redacted_thinking":
                    self._last_thinking_blocks.append(block)

                elif block_type == "text":
                    content += block.get("text", "")

                elif block_type == "tool_use":
                    tool_calls.append(ToolCall(
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                        raw_xml=f'<mesh_call name="{block.get("name", "")}"></mesh_call>',
                        call_id=block.get("id", ""),
                    ))

            self._last_reasoning_content = None
            thinking_text = ""
            for tb in self._last_thinking_blocks:
                if tb.get("type") == "thinking":
                    thinking_text += tb.get("thinking", "")
            if thinking_text.strip():
                self._last_reasoning_content = thinking_text.strip()

            usage_data = data.get("usage", {})
            self._last_usage = {
                "input_tokens": usage_data.get("input_tokens", 0),
                "output_tokens": usage_data.get("output_tokens", 0),
                "cache_creation_tokens": usage_data.get("cache_creation_input_tokens", 0),
                "cache_read_tokens": usage_data.get("cache_read_input_tokens", 0),
                "reasoning_tokens": 0,
                "total_tokens": usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
                "backend": "anthropic",
                "model": model,
            }

            logger.debug(
                "LLM multi-turn-anthropic response: len=%d tool_calls=%d "
                "input_tokens=%s output_tokens=%s",
                len(content), len(tool_calls),
                usage_data.get("input_tokens", "?"),
                usage_data.get("output_tokens", "?"),
            )

            return content, tool_calls, dict(self._last_usage)

        return await self._retry_with_backoff(_make_request, f"Multi-turn-anthropic ({model})")

    async def _complete_claude_code(
        self,
        prompt: str,
        model: str,
        callback: LLMStreamCallback | None = None,
        env_override: dict[str, str] | None = None,
        mcp_config: str | None = None,
        cc_system_prompt: str | None = None,
        cc_watchdog: Any = None,
    ) -> str:
        """Complete using Claude Code subprocess with streaming tool visibility.

        Supports multi-account fallback: if cc_fallback_homes is configured,
        retries with alternate HOME dirs (each pointing to a different CC account)
        when the primary account fails (e.g., rate-limited, overloaded).

        Args:
            prompt: The prompt to send
            model: Model name
            callback: Optional callback for tool events
            env_override: Optional environment variable overrides (used by _complete_zai)

        Returns:
            The final response text
        """
        if not shutil.which("claude"):
            raise RuntimeError("Claude CLI not found. Is it installed and in PATH?")

        # Build the list of HOME dirs to try: default (None) + fallbacks
        # Filter out disabled accounts (.disabled marker in .claude dir)
        all_homes: list[str | None] = []
        import pwd
        real_home = pwd.getpwuid(os.getuid()).pw_dir
        for h in [None] + [h.replace("~", real_home, 1) if h.startswith("~") else h for h in self.config.cc_fallback_homes]:
            if h is None:
                marker = os.path.join(real_home, ".claude", ".disabled")
            else:
                marker = os.path.join(h, ".claude", ".disabled")
            if os.path.exists(marker):
                logger.debug(f"CC account {h or 'default'} is disabled, skipping")
                continue
            all_homes.append(h)

        if not all_homes:
            logger.error("All CC accounts are disabled — no account available")
            all_homes = [None]  # last resort: try default anyway

        n = len(all_homes)

        # Load-balance: start from next account in rotation (spreads usage
        # across accounts and lets idle accounts refresh their OAuth tokens)
        import time as _time
        now = _time.monotonic()

        # Purge expired cooldowns
        self._cc_depleted = {k: v for k, v in self._cc_depleted.items() if v > now}

        # Build try-order: non-depleted first (round-robin), then depleted as fallback
        start_idx = self._cc_home_idx % n
        self._cc_home_idx = (self._cc_home_idx + 1) % n  # advance for next call

        available = []
        depleted = []
        for i in range(n):
            idx = (start_idx + i) % n
            if idx in self._cc_depleted:
                depleted.append(idx)
            else:
                available.append(idx)
        try_order = available + depleted  # try healthy accounts first

        if depleted:
            depleted_labels = [all_homes[i] or "default" for i in depleted]
            logger.info(f"CC load-balancer: skipping depleted accounts {depleted_labels}, "
                        f"{len(available)} available")

        last_error: RuntimeError | None = None

        for attempt, idx in enumerate(try_order):
            home_dir = all_homes[idx]
            try:
                result = await self._run_cc_subprocess(
                    prompt, model, callback, env_override, home_override=home_dir,
                    mcp_config=mcp_config,
                    cc_system_prompt=cc_system_prompt,
                    cc_watchdog=cc_watchdog,
                )
                # Success — clear any cooldown for this account
                self._cc_depleted.pop(idx, None)
                return result
            except RuntimeError as e:
                last_error = e
                home_label = home_dir or "default"
                # Mark this account as depleted for 5 minutes
                self._cc_depleted[idx] = now + 300
                logger.info(f"CC account {home_label} marked depleted for 5 min")

                if attempt < len(try_order) - 1:
                    next_idx = try_order[attempt + 1]
                    next_home = all_homes[next_idx]
                    next_label = next_home or "default"
                    # Fetch usage for the failed account to show why it failed
                    failed_usage = self._fetch_cc_account_usage(home_dir)
                    next_usage = self._fetch_cc_account_usage(next_home)
                    switch_msg = (
                        f"CC account switch: {home_label} failed → trying {next_label}\n"
                        f"  {home_label}: {failed_usage}\n"
                        f"  {next_label}: {next_usage}"
                    )
                    logger.warning(switch_msg)
                    # Emit as a tool event so the user sees it in their activity feed
                    if callback:
                        callback.on_cc_tool_event(CCToolEvent(
                            event_type="tool_result",
                            call_id=f"cc-acct-switch-{attempt}",
                            tool_name="cc:AccountSwitch",
                            data=switch_msg,
                        ))
                else:
                    logger.error(f"CC failed on all {len(try_order)} accounts (last HOME={home_label}): {e}")

        raise last_error  # type: ignore[misc]

    @staticmethod
    def _fetch_cc_account_usage(home_dir: str | None) -> str:
        """Fetch CC account usage summary for a given HOME dir. Returns a compact string."""
        from pathlib import Path
        try:
            import httpx
        except ImportError:
            return "usage unavailable (httpx not installed)"

        home = Path(home_dir) if home_dir else Path.home()
        creds_path = home / ".claude" / ".credentials.json"
        if not creds_path.exists():
            return "no credentials found"

        try:
            creds = json.loads(creds_path.read_text())
        except Exception:
            return "credentials unreadable"

        oauth = creds.get("claudeAiOauth")
        if not oauth or not isinstance(oauth, dict):
            return "no OAuth credentials"

        access_token = oauth.get("accessToken", "")
        if not access_token:
            return "no access token"

        # Refresh token if expired
        import time as _time
        refresh_token = oauth.get("refreshToken", "")
        expires_at_ms = oauth.get("expiresAt", 0)
        now_ms = int(_time.time() * 1000)
        if expires_at_ms > 0 and now_ms > expires_at_ms - 600_000 and refresh_token:
            try:
                r = httpx.post(
                    "https://api.anthropic.com/v1/oauth/token",
                    json={"grant_type": "refresh_token", "refresh_token": refresh_token},
                    headers={"Content-Type": "application/json"},
                    timeout=5,
                )
                if r.status_code == 200:
                    token_data = r.json()
                    access_token = token_data.get("access_token", access_token)
            except Exception:
                pass

        try:
            r = httpx.get(
                "https://api.anthropic.com/api/oauth/usage",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "anthropic-beta": "oauth-2025-04-20",
                },
                timeout=5,
            )
            if r.status_code != 200:
                return f"HTTP {r.status_code}"
            data = r.json()
        except Exception as e:
            return f"fetch failed: {e}"

        # Build compact summary: "5h: 45.2% | 7d: 72.1% | opus: 68.3%"
        parts = []
        label_map = {
            "five_hour": "5h",
            "seven_day": "7d",
            "seven_day_opus": "opus",
            "seven_day_sonnet": "sonnet",
        }
        for key, label in label_map.items():
            info = data.get(key)
            if info and info.get("utilization") is not None:
                util = info["utilization"]
                parts.append(f"{label}: {util:.1f}%")
        return " | ".join(parts) if parts else "no usage data"

    async def _run_cc_subprocess(
        self,
        prompt: str,
        model: str,
        callback: LLMStreamCallback | None = None,
        env_override: dict[str, str] | None = None,
        home_override: str | None = None,
        mcp_config: str | None = None,
        cc_system_prompt: str | None = None,
        cc_watchdog: Any = None,
    ) -> str:
        """Run a single Claude Code subprocess attempt.

        Args:
            prompt: The prompt to send
            model: Model name
            callback: Optional callback for tool events
            env_override: Optional environment variable overrides
            home_override: Optional HOME directory override for multi-account fallback

        Returns:
            The final response text
        """
        home_label = home_override or "default"
        logger.debug(
            f"LLM request [claude-code]: model={model} prompt_len={len(prompt)} HOME={home_label}"
        )
        logger.debug(f"LLM prompt preview (first 500 chars): {prompt[:500]!r}...")

        # Apply explicit thinking mode via model name prefix.
        # CC auto-detects thinking from the model name: "openai/*" → thinking ON.
        # When cc_thinking is explicitly set, we transform the model name to match.
        effective_model = model
        if self.config.cc_thinking is True:
            if not model.startswith("openai/"):
                effective_model = f"openai/{model}"
                logger.debug(f"CC thinking=true: model {model!r} → {effective_model!r}")
        elif self.config.cc_thinking is False:
            if model.startswith("openai/"):
                effective_model = model[len("openai/"):]
                logger.debug(f"CC thinking=false: model {model!r} → {effective_model!r}")

        # Build command — use configured binary or auto-detect
        claude_bin = self.config.cc_binary or shutil.which("claude") or "claude"
        cmd = [
            claude_bin, "-p",
            "--model", effective_model,
            "--output-format", "stream-json",
            "--verbose",
        ]

        safe_mcp_only = bool(self.config.cc_safe_mcp_only)
        if safe_mcp_only:
            # Disable Claude's filesystem/shell/agent surface.  The only tools
            # permitted without prompting are the named tools from the strict
            # per-invocation MCP configuration.  `dontAsk` makes any attempted
            # escape fail closed in non-interactive mode.
            cmd.extend(["--tools", ""])
            cmd.extend(["--permission-mode", "dontAsk"])
            cmd.extend(["--no-session-persistence"])
            if mcp_config:
                if not self.config.cc_safe_mcp_tools:
                    raise RuntimeError(
                        "cc_safe_mcp_only with MCP requires a non-empty tool allowlist"
                    )
                cmd.extend(["--strict-mcp-config"])
                cmd.extend([
                    "--allowedTools", ",".join(self.config.cc_safe_mcp_tools),
                ])
            elif self.config.cc_safe_mcp_tools:
                raise RuntimeError(
                    "cc_safe_mcp_tools were configured without an MCP configuration"
                )
        elif self.config.cc_allowed_tools:
            cmd.extend(["--allowedTools", ",".join(self.config.cc_allowed_tools)])

        # Set effort level (always high for all CC calls)
        if self.cc_effort:
            cmd.extend(["--effort", self.cc_effort])

        # Existing interactive-agent behavior is unchanged.  Restricted batch
        # jobs use the MCP-only branch above and must never receive this flag.
        if not safe_mcp_only:
            cmd.append("--dangerously-skip-permissions")

        # MCP config: inject mesh tools as native MCP tools for CC
        if mcp_config:
            cmd.extend(["--mcp-config", mcp_config])

        # System prompt: passed separately for CC compaction durability
        if cc_system_prompt:
            cmd.extend(["--system-prompt", cc_system_prompt])

        # Session persistence enabled — agent CC calls save JSONL traces

        # Build curated environment — allowlist blocks LD_PRELOAD etc.
        proc_env = _build_subprocess_env()
        proc_env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        proc_env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
        if self.config.node_id:
            proc_env["MESH_NODE_ID"] = self.config.node_id
        # Ensure mesh-tool CLI is on PATH (lives in repo root)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in proc_env.get("PATH", "").split(os.pathsep):
            proc_env["PATH"] = repo_root + os.pathsep + proc_env.get("PATH", "")
        # Intentional allowlist bypass: cc_env and env_override are operator-controlled
        # config values, not inherited from the process environment. They may inject
        # arbitrary vars (e.g., account-specific HOME, custom API keys) after filtering.
        if self.config.cc_env:
            proc_env.update(self.config.cc_env)
        if env_override:
            proc_env.update(env_override)
        if home_override:
            # Preserve the real user's ~/.local/bin in PATH so CC finds its binary
            # Use pwd.getpwuid to read /etc/passwd — immune to HOME env overrides
            real_home = pwd.getpwuid(os.getuid()).pw_dir
            real_local_bin = os.path.join(real_home, ".local", "bin")
            path = proc_env.get("PATH", "")
            if real_local_bin not in path.split(os.pathsep):
                proc_env["PATH"] = real_local_bin + os.pathsep + path
            proc_env["HOME"] = home_override

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=proc_env,
                start_new_session=True,  # new process group for clean cleanup
            )
            logger.debug(f"CC subprocess started: pid={proc.pid} pgid={proc.pid}")

            # Send prompt via stdin
            proc.stdin.write(prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()

            # Stream and process events
            final_result = ""
            call_id_to_name: dict[str, str] = {}
            # Accumulate token usage across all assistant events
            cc_usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 0, "cache_read_tokens": 0}
            # Accumulate text blocks from assistant events so they aren't lost
            # when CC's result event is only a short summary
            cc_text_blocks: list[str] = []

            # CC Watchdog: segment streaming events into logical turns and
            # periodically invoke an assessor LLM to evaluate progress.
            _wdog_turn_count = 0
            _wdog_current_turn: list[dict] = []
            _wdog_all_turns: list[list[dict]] = []
            _wdog_task: asyncio.Task | None = None
            _wdog_killed = False

            # Read stdout in chunks and split by newlines
            # (readline has a 64KB limit which Claude Code can exceed)
            idle_timeout = self.config.cc_idle_timeout or None  # 0 → disabled
            first_output_received = False
            buffer = b""
            while True:
                try:
                    if idle_timeout and not first_output_received:
                        chunk = await asyncio.wait_for(
                            proc.stdout.read(1024 * 1024), timeout=idle_timeout,
                        )
                    else:
                        chunk = await proc.stdout.read(1024 * 1024)  # 1MB chunks
                except asyncio.TimeoutError:
                    logger.warning(
                        f"CC subprocess initial timeout ({idle_timeout}s) — "
                        f"no first output for {idle_timeout}s, killing pid={proc.pid}"
                    )
                    raise RuntimeError(
                        f"Claude Code subprocess initial timeout: no output for {idle_timeout}s"
                    )
                if not chunk:
                    # Process any remaining data in buffer
                    if buffer.strip():
                        try:
                            event = json.loads(buffer.decode())
                            final_result = self._process_cc_event(
                                event, callback, call_id_to_name, final_result,
                                cc_usage, cc_text_blocks,
                            )
                        except json.JSONDecodeError:
                            pass
                    break

                if not first_output_received:
                    first_output_received = True
                    logger.debug(f"CC subprocess pid={proc.pid} produced first output")
                buffer += chunk

                # Process complete lines
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode().strip()
                    if not line_str:
                        continue

                    try:
                        event = json.loads(line_str)
                        final_result = self._process_cc_event(
                            event, callback, call_id_to_name, final_result,
                            cc_usage, cc_text_blocks,
                        )
                    except json.JSONDecodeError:
                        continue

                    # --- CC Watchdog: turn segmentation & progress evaluation ---
                    if cc_watchdog:
                        evt_type = event.get("type")
                        if evt_type == "assistant":
                            msg = event.get("message", {})
                            if isinstance(msg, dict):
                                for blk in (msg.get("content") or []):
                                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                                        _wdog_current_turn.append({
                                            "type": "call",
                                            "name": blk.get("name", "?"),
                                            "input": str(blk.get("input", {})),
                                        })
                        elif evt_type == "user":
                            msg = event.get("message", {})
                            if isinstance(msg, dict):
                                for blk in (msg.get("content") or []):
                                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                                        cid = blk.get("tool_use_id", "")
                                        _wdog_current_turn.append({
                                            "type": "result",
                                            "name": call_id_to_name.get(cid, "?"),
                                            "output": str(blk.get("content", "")),
                                        })
                            if _wdog_current_turn:
                                _wdog_turn_count += 1
                                _wdog_all_turns.append(_wdog_current_turn)
                                _wdog_current_turn = []

                                if _wdog_task is not None and _wdog_task.done():
                                    try:
                                        if _wdog_task.result():
                                            logger.warning(
                                                "CC watchdog: kill signal at turn %d, pid=%d",
                                                _wdog_turn_count, proc.pid,
                                            )
                                            _wdog_killed = True
                                            try:
                                                os.killpg(proc.pid, signal.SIGTERM)
                                            except (ProcessLookupError, PermissionError):
                                                pass
                                            break
                                    except Exception:
                                        logger.debug("CC watchdog task raised, ignoring")
                                    _wdog_task = None

                                if (
                                    _wdog_turn_count > 0
                                    and _wdog_turn_count % 8 == 0
                                    and _wdog_task is None
                                ):
                                    logger.info(
                                        "CC watchdog: firing evaluation at turn %d, pid=%d",
                                        _wdog_turn_count, proc.pid,
                                    )
                                    _wdog_task = asyncio.create_task(
                                        cc_watchdog(list(_wdog_all_turns))
                                    )

                if _wdog_killed:
                    break

            # Handle watchdog kill before normal completion
            if _wdog_killed:
                if _wdog_task and not _wdog_task.done():
                    _wdog_task.cancel()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                accumulated = "\n\n".join(cc_text_blocks) if cc_text_blocks else final_result
                kill_msg = (
                    f"[CC watchdog terminated executor after {_wdog_turn_count} turns: "
                    f"consecutive progress checks returned NO.]"
                )
                logger.info("CC watchdog kill complete: %s", kill_msg)
                proc = None
                return f"{accumulated}\n\n{kill_msg}" if accumulated else kill_msg

            # Cancel any pending watchdog task on normal completion
            if _wdog_task and not _wdog_task.done():
                _wdog_task.cancel()

            # Wait for process to complete
            await proc.wait()

            if proc.returncode != 0:
                stderr_data = await proc.stderr.read()
                error_msg = stderr_data.decode().strip() if stderr_data else f"Exit code {proc.returncode}"
                raise RuntimeError(f"Claude Code error: {error_msg}")

            # If CC's result is short but it produced substantial intermediate
            # text blocks, prepend the accumulated text.  This recovers analysis
            # that CC wrote between tool calls but didn't repeat in the result.
            # Skip if the result already contains mesh_call (send_message) —
            # those are handled by the tool-call parser and we don't want to
            # duplicate content.
            accumulated_text = "\n\n".join(cc_text_blocks)
            if (
                cc_text_blocks
                and len(accumulated_text) > len(final_result) * 2
                and "mesh_call" not in final_result
            ):
                logger.info(
                    f"CC text recovery: {len(cc_text_blocks)} blocks "
                    f"({len(accumulated_text)} chars) vs result ({len(final_result)} chars)"
                )
                final_result = accumulated_text + "\n\n" + final_result
            elif cc_text_blocks:
                logger.debug(
                    f"CC text blocks present ({len(cc_text_blocks)} blocks, "
                    f"{len(accumulated_text)} chars) but result is sufficient "
                    f"({len(final_result)} chars) — not prepending"
                )

            # Store cumulative usage from all assistant events
            self._last_usage = {
                **cc_usage,
                "reasoning_tokens": 0,
                "total_tokens": cc_usage["input_tokens"] + cc_usage["output_tokens"],
                "backend": "claude-code",
                "model": model,
                "cc_home": home_label,
            }

            logger.debug(f"LLM response: len={len(final_result)} HOME={home_label}")
            logger.debug(f"LLM response content preview: {final_result[:500]!r}...")
            proc = None  # success — don't kill in finally
            return final_result

        except asyncio.TimeoutError:
            raise RuntimeError("Claude Code subprocess timed out")
        finally:
            if proc is not None and proc.returncode is None:
                # Subprocess still running — kill it and its entire process group
                pgid = proc.pid
                logger.warning(
                    f"CC subprocess cleanup: killing pid={proc.pid} pgid={pgid} "
                    f"(cancelled or errored)"
                )
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                logger.info(f"CC subprocess cleanup complete: pid={proc.pid}")

    def _process_cc_event(
        self,
        event: dict,
        callback: LLMStreamCallback | None,
        call_id_to_name: dict[str, str],
        final_result: str,
        cc_usage: dict | None = None,
        cc_text_blocks: list[str] | None = None,
    ) -> str:
        """Process a single Claude Code JSON event and emit callbacks.

        Returns updated final_result.

        If cc_text_blocks is provided, text blocks from assistant events are
        accumulated into it so the caller can include them in the final result
        when CC's ``result`` event is only a short summary.
        """
        event_type = event.get("type")

        if event_type == "assistant":
            msg = event.get("message", {})
            if not isinstance(msg, dict):
                return final_result

            # Extract and accumulate token usage from assistant events
            if cc_usage is not None:
                usage = msg.get("usage", {})
                if isinstance(usage, dict):
                    cc_usage["input_tokens"] += usage.get("input_tokens", 0)
                    cc_usage["output_tokens"] += usage.get("output_tokens", 0)
                    cc_usage["cache_creation_tokens"] += usage.get("cache_creation_input_tokens", 0)
                    cc_usage["cache_read_tokens"] += usage.get("cache_read_input_tokens", 0)

            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "").strip()
                        if text and cc_text_blocks is not None:
                            cc_text_blocks.append(text)

                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        call_id = block.get("id", f"cc-{uuid.uuid4().hex[:8]}")
                        tool_name = block.get("name", "unknown")
                        tool_input = block.get("input", {})

                        # Track for result matching
                        call_id_to_name[call_id] = tool_name

                        # Handle TodoWrite specially
                        if tool_name == "TodoWrite" and callback:
                            todos = tool_input.get("todos", [])
                            if hasattr(callback, "on_todos"):
                                callback.on_todos(todos)

                        # Emit tool_call event
                        if callback and hasattr(callback, "on_cc_tool_event"):
                            callback.on_cc_tool_event(CCToolEvent(
                                event_type="tool_call",
                                call_id=call_id,
                                tool_name=f"cc:{tool_name}",
                                data=tool_input,
                            ))

        elif event_type == "user":
            msg = event.get("message", {})
            if not isinstance(msg, dict):
                return final_result
            content = msg.get("content", [])

            # Check for todos in tool_use_result
            tool_use_result = event.get("tool_use_result")
            if isinstance(tool_use_result, dict) and tool_use_result.get("newTodos"):
                if callback and hasattr(callback, "on_todos"):
                    callback.on_todos(tool_use_result["newTodos"])

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        call_id = block.get("tool_use_id", "")
                        result_content = block.get("content", "")
                        tool_name = call_id_to_name.get(call_id, "unknown")

                        # Emit tool_result event
                        if callback and hasattr(callback, "on_cc_tool_event"):
                            callback.on_cc_tool_event(CCToolEvent(
                                event_type="tool_result",
                                call_id=call_id,
                                tool_name=f"cc:{tool_name}",
                                data=result_content,
                            ))

        elif event_type == "result":
            final_result = event.get("result", "")
            # Strip hallucinated user turns (Claude Code sometimes continues the
            # conversation by hallucinating "Human:" messages after its response)
            final_result = _strip_hallucinated_turns(final_result)

        return final_result

    async def _complete_zai(
        self,
        prompt: str,
        model: str,
        callback: LLMStreamCallback | None = None,
        mcp_config: str | None = None,
        cc_system_prompt: str | None = None,
        cc_watchdog: Any = None,
    ) -> str:
        """Complete using Z.AI via Claude Code subprocess.

        LEGACY: This is a special-cased version of _complete_claude_code with
        hardcoded Z.AI env vars. New integrations should use the generic cc_env
        config on a claude-code backend instead. This method is kept for backward
        compatibility with existing 'zai' backend_type entries.
        """
        if not self.config.zai_api_key:
            raise RuntimeError("ZAI_API_KEY not set")

        logger.debug(
            f"LLM request [zai]: model={model} prompt_len={len(prompt)}"
        )

        # Build Z.AI environment overrides
        env_override = {
            "ANTHROPIC_AUTH_TOKEN": self.config.zai_api_key,
            "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }

        # Delegate to _complete_claude_code with env override
        result = await self._complete_claude_code(
            prompt, model, callback=callback, env_override=env_override,
            mcp_config=mcp_config,
            cc_system_prompt=cc_system_prompt,
            cc_watchdog=cc_watchdog,
        )
        # Override backend label for Z.AI
        if self._last_usage:
            self._last_usage["backend"] = "zai"
        return result

    async def _complete_codex(
        self,
        prompt: str,
        model: str,
        callback: LLMStreamCallback | None = None,
    ) -> str:
        """Complete using Codex CLI subprocess.

        Codex has its own built-in tools and internal tool loop.
        Mesh-provided extended tools (email, calendar, notes, etc.) are NOT
        available through this backend — Codex agents are limited to Codex's
        built-in surface (apply_patch, shell, list_dir, etc.).  MCP integration
        would be needed to bridge mesh tools into Codex; that's future work.
        """
        codex_bin = (
            self.config.codex_binary
            or shutil.which("codex")
            or "codex"
        )

        # Pass prompt via stdin (codex reads from stdin when prompt is "-")
        # to avoid OS argv length limit (E2BIG) on large prompts.
        cmd = [
            codex_bin, "exec", "-",
            "-m", model,
            "--ephemeral",
            "--json",
        ]

        extra_args = list(self.config.codex_extra_args or [])
        # `--sandbox` is mutually exclusive with `--dangerously-bypass-approvals-and-sandbox`.
        # If the caller specifies a sandbox mode, omit the bypass default.
        if not any(a == "--sandbox" or a.startswith("--sandbox=") for a in extra_args):
            cmd.append("--dangerously-bypass-approvals-and-sandbox")

        effort = self.config.cc_effort
        if effort:
            cmd.extend(["-c", f'model_reasoning_effort="{effort}"'])

        if extra_args:
            cmd.extend(extra_args)

        # Codex requires real HOME for ~/.codex/auth.json resolution.
        # CC account rotation sets HOME to ~/.claude-acctN which breaks Codex.
        # Use pwd.getpwuid to read /etc/passwd — immune to HOME env overrides.
        proc_env = _build_subprocess_env()
        real_home = pwd.getpwuid(os.getuid()).pw_dir
        proc_env["HOME"] = real_home
        if self.config.node_id:
            proc_env["MESH_NODE_ID"] = self.config.node_id
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc_env["PATH"] = (
            repo_root + ":"
            + os.path.join(real_home, ".local/share/node-v22/bin") + ":"
            + os.path.join(real_home, ".local/bin") + ":"
            + proc_env.get("PATH", "")
        )

        logger.debug(f"Codex subprocess: {codex_bin} exec ... -m {model}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
            start_new_session=True,
        )

        if proc.stdin:
            try:
                proc.stdin.write(prompt.encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            proc.stdin.close()

        codex_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "cached_input_tokens": 0,
        }
        text_blocks: list[str] = []
        buffer = b""

        idle_timeout = self.config.codex_subprocess_idle_timeout
        try:
            while True:
                try:
                    if idle_timeout and idle_timeout > 0:
                        chunk = await asyncio.wait_for(
                            proc.stdout.read(1024 * 1024),
                            timeout=idle_timeout,
                        )
                    else:
                        chunk = await proc.stdout.read(1024 * 1024)
                except asyncio.TimeoutError:
                    logger.warning(f"Codex subprocess timeout — no output for {idle_timeout}s")
                    raise RuntimeError(f"Codex subprocess timeout: no output for {idle_timeout}s")

                if not chunk:
                    if buffer.strip():
                        try:
                            event = json.loads(buffer.decode())
                            self._process_codex_event(event, codex_usage, text_blocks, callback)
                        except json.JSONDecodeError:
                            pass
                    break

                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode().strip()
                    if not line_str:
                        continue
                    try:
                        event = json.loads(line_str)
                        self._process_codex_event(event, codex_usage, text_blocks, callback)
                    except json.JSONDecodeError:
                        continue

            await proc.wait()

            if proc.returncode != 0:
                stderr_data = await proc.stderr.read()
                error_msg = stderr_data.decode().strip() if stderr_data else f"Exit code {proc.returncode}"
                logger.warning(f"Codex exited {proc.returncode}: {error_msg[:500]}")

            final_text = "\n\n".join(text_blocks)

            self._last_usage = {
                **codex_usage,
                "total_tokens": codex_usage["input_tokens"] + codex_usage["output_tokens"],
                "backend": "codex",
                "model": model,
            }

            logger.debug(f"Codex response: len={len(final_text)}")
            return final_text

        finally:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass

    def _process_codex_event(
        self,
        event: dict,
        usage: dict,
        text_blocks: list[str],
        callback: LLMStreamCallback | None,
    ) -> None:
        """Process a single Codex JSONL event."""
        event_type = event.get("type")

        # Codex emits dotted event types: thread.started, turn.started,
        # item.completed, turn.completed. Older shape used underscores; accept both.
        if event_type in ("turn.completed", "turn_complete"):
            turn_usage = event.get("usage", {})
            if isinstance(turn_usage, dict):
                usage["input_tokens"] += turn_usage.get("input_tokens", 0)
                usage["output_tokens"] += turn_usage.get("output_tokens", 0)
                usage["reasoning_output_tokens"] += turn_usage.get("reasoning_output_tokens", 0)
                usage["cached_input_tokens"] += turn_usage.get("cached_input_tokens", 0)

        elif event_type == "agent_message":
            message = event.get("message", "")
            if isinstance(message, str) and message.strip():
                text_blocks.append(message.strip())

        elif event_type in (
            "item.started", "item_started",
            "item.completed", "item_completed",
        ):
            item = event.get("item", {})
            if not isinstance(item, dict):
                return
            # Codex item.completed wraps payloads of various types; only
            # agent_message items carry the assistant's text reply.
            item_type = item.get("type")
            if (
                event_type in ("item.completed", "item_completed")
                and item_type in (None, "agent_message")
            ):
                text = item.get("text", "")
                if isinstance(text, str) and text.strip():
                    text_blocks.append(text.strip())
                return

            codex_tool_event = self._codex_item_tool_event(event_type, item)
            if codex_tool_event and callback and hasattr(callback, "on_cc_tool_event"):
                callback.on_cc_tool_event(codex_tool_event)

        elif event_type == "exec_command_begin":
            if callback and hasattr(callback, "on_cc_tool_event"):
                callback.on_cc_tool_event(CCToolEvent(
                    event_type="tool_call",
                    call_id=event.get("call_id", f"codex-{uuid.uuid4().hex[:8]}"),
                    tool_name="codex:shell",
                    data={"command": event.get("command", "")},
                ))

        elif event_type == "exec_command_end":
            if callback and hasattr(callback, "on_cc_tool_event"):
                callback.on_cc_tool_event(CCToolEvent(
                    event_type="tool_result",
                    call_id=event.get("call_id", f"codex-{uuid.uuid4().hex[:8]}"),
                    tool_name="codex:shell",
                    data={"exit_code": event.get("exit_code"), "duration": event.get("duration")},
                ))

        elif event_type == "patch_apply_begin":
            if callback and hasattr(callback, "on_cc_tool_event"):
                callback.on_cc_tool_event(CCToolEvent(
                    event_type="tool_call",
                    call_id=event.get("call_id", f"codex-{uuid.uuid4().hex[:8]}"),
                    tool_name="codex:patch",
                    data={},
                ))

        elif event_type == "patch_apply_end":
            if callback and hasattr(callback, "on_cc_tool_event"):
                callback.on_cc_tool_event(CCToolEvent(
                    event_type="tool_result",
                    call_id=event.get("call_id", f"codex-{uuid.uuid4().hex[:8]}"),
                    tool_name="codex:patch",
                    data={"duration": event.get("duration")},
                ))

        elif event_type == "mcp_tool_call_begin":
            if callback and hasattr(callback, "on_cc_tool_event"):
                callback.on_cc_tool_event(CCToolEvent(
                    event_type="tool_call",
                    call_id=event.get("call_id", f"codex-{uuid.uuid4().hex[:8]}"),
                    tool_name=f"codex:mcp:{event.get('name', 'unknown')}",
                    data=event.get("arguments", {}),
                ))

        elif event_type == "mcp_tool_call_end":
            if callback and hasattr(callback, "on_cc_tool_event"):
                callback.on_cc_tool_event(CCToolEvent(
                    event_type="tool_result",
                    call_id=event.get("call_id", f"codex-{uuid.uuid4().hex[:8]}"),
                    tool_name=f"codex:mcp:{event.get('name', 'unknown')}",
                    data=event.get("result", {}),
                ))

    def _codex_item_tool_event(
        self,
        event_type: str,
        item: dict,
    ) -> CCToolEvent | None:
        """Translate current Codex JSONL item events into mesh tool activity."""
        item_type = item.get("type")
        if not isinstance(item_type, str):
            return None

        is_start = event_type in ("item.started", "item_started")
        is_done = event_type in ("item.completed", "item_completed")
        if not is_start and not is_done:
            return None

        call_id = str(
            item.get("id")
            or item.get("call_id")
            or f"codex-{uuid.uuid4().hex[:8]}"
        )

        if item_type == "command_execution":
            command = item.get("command", "")
            if is_start:
                return CCToolEvent(
                    event_type="tool_call",
                    call_id=call_id,
                    tool_name="codex:shell",
                    data={"command": command},
                )

            data: dict[str, Any] = {
                "command": command,
                "exit_code": item.get("exit_code"),
                "status": item.get("status"),
            }
            output = item.get("aggregated_output")
            if output is not None:
                data["output"] = output
            return CCToolEvent(
                event_type="tool_result",
                call_id=call_id,
                tool_name="codex:shell",
                data=data,
            )

        if item_type in ("patch_apply", "patch_application", "apply_patch"):
            if is_start:
                data = {
                    k: v for k, v in item.items()
                    if k not in ("id", "type", "status")
                }
                return CCToolEvent(
                    event_type="tool_call",
                    call_id=call_id,
                    tool_name="codex:patch",
                    data=data,
                )

            return CCToolEvent(
                event_type="tool_result",
                call_id=call_id,
                tool_name="codex:patch",
                data={
                    "status": item.get("status"),
                    "exit_code": item.get("exit_code"),
                    "output": item.get("aggregated_output") or item.get("output") or "",
                },
            )

        if item_type in ("mcp_tool_call", "mcp_tool_execution", "tool_call"):
            name = (
                item.get("name")
                or item.get("tool_name")
                or item.get("tool")
                or "unknown"
            )
            tool_name = (
                f"codex:mcp:{name}"
                if item_type.startswith("mcp_")
                else f"codex:{name}"
            )
            if is_start:
                return CCToolEvent(
                    event_type="tool_call",
                    call_id=call_id,
                    tool_name=tool_name,
                    data=item.get("arguments") or item.get("input") or {},
                )

            return CCToolEvent(
                event_type="tool_result",
                call_id=call_id,
                tool_name=tool_name,
                data=item.get("result") or item.get("output") or item,
            )

        return None

    async def _complete_claude_interactive(
        self,
        prompt: str,
        model: str,
        timeout: int = 10800,
    ) -> str:
        """Complete using claude_interactive.py (tmux-based interactive wrapper).

        Shells out to the standalone module — prompt on stdin, response on stdout.
        Every call automatically gets a full session transcript via --log-file.
        """
        ci_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "claude_interactive.py",
        )
        cc_bin = self.config.cc_binary or shutil.which("claude") or "claude"
        effort = self.config.cc_effort or "high"

        # Auto-generate log path so every interactive call gets a transcript
        session_name = f"ci-{uuid.uuid4().hex[:8]}"
        log_file = f"/tmp/ci-{session_name}.log"
        self._ci_log_file = log_file  # Plumbed for worker_status visibility

        cmd = [sys.executable, ci_script]
        if model:
            cmd.extend(["--model", model])
        if effort:
            cmd.extend(["--effort", effort])
        cmd.extend(["--permission-mode", "agentic"])
        cmd.extend(["--timeout", str(timeout)])
        cmd.extend(["--log-file", log_file])
        cmd.extend(["--session-name", session_name])
        if cc_bin:
            cmd.extend(["--cc-binary", cc_bin])

        real_home = pwd.getpwuid(os.getuid()).pw_dir
        proc_env = _build_subprocess_env()
        proc_env["HOME"] = real_home
        if self.config.node_id:
            proc_env["MESH_NODE_ID"] = self.config.node_id
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc_env["PATH"] = (
            repo_root + ":"
            + os.path.join(real_home, ".local/share/node-v22/bin") + ":"
            + os.path.join(real_home, ".local/bin") + ":"
            + proc_env.get("PATH", "")
        )

        logger.debug(f"Claude interactive subprocess: {ci_script} --model {model}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )

        try:
            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    proc.communicate(input=prompt.encode("utf-8")),
                    timeout=timeout + 30,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(
                    f"Claude interactive timed out (model={model}, timeout={timeout}s)"
                )

            result = stdout_data.decode("utf-8", errors="replace").strip()

            if proc.returncode == 2:
                raise TimeoutError(
                    f"Claude interactive timed out (model={model})"
                )
            elif proc.returncode != 0:
                detail = stderr_data.decode("utf-8", errors="replace").strip()[-500:]
                raise RuntimeError(
                    f"Claude interactive failed (exit {proc.returncode}): {detail}"
                )

            self._last_usage = {
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(result) // 4,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": (len(prompt) + len(result)) // 4,
                "backend": "claude-interactive",
                "model": model,
            }

            return result
        finally:
            self._ci_log_file = None

    async def _complete_mesh_harness(
        self,
        prompt: str,
        model: str,
        callback: LLMStreamCallback | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Complete using the mesh standalone harness subprocess.

        Runs `python -m mesh.harness exec` with the configured sub-backend
        (e.g., anthropic via z.ai).  The harness has its own TAOR loop and
        tool surface.  Tool set is controlled by harness_toolset config:
        "legacy" gives the full mesh tool set, "harness" gives the restricted
        Codex-style 4-tool surface.
        """
        python = self.config.harness_python or sys.executable or "python3"

        # Pass prompt via stdin ("--prompt -") to avoid OS argv length limit
        # (E2BIG / errno 7) on large prompts. The harness CLI reads stdin when
        # --prompt is "-" or omitted.
        cmd = [
            python, "-m", "mesh.harness", "exec",
            "--backend", self.config.harness_backend or "anthropic",
            "--model", model,
            "--prompt", "-",
        ]

        effort = self.config.cc_effort
        if effort:
            cmd.extend(["--effort", effort])
        if self.config.thinking_budget:
            cmd.extend(["--thinking-budget", str(self.config.thinking_budget)])
        if self.config.harness_base_url:
            cmd.extend(["--base-url", self.config.harness_base_url])

        api_key = self.config.harness_api_key
        if not api_key and self.config.harness_backend == "anthropic":
            api_key = self.config.anthropic_api_key or self.config.api_key
        if api_key:
            cmd.extend(["--api-key", api_key])

        if self.config.harness_tools:
            cmd.extend(["--tools", self.config.harness_tools])
        elif self.config.harness_toolset:
            cmd.extend(["--toolset", self.config.harness_toolset])

        sp_file = self.config.harness_system_prompt_file
        if not sp_file:
            # Default to bundled harness system prompt
            sp_file = os.path.join(os.path.dirname(__file__), "harness", "system_prompt.md")
            if not os.path.isfile(sp_file):
                sp_file = ""
        if sp_file:
            cmd.extend(["--system-prompt-file", sp_file])
        elif system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        if self.config.harness_agent_socket:
            cmd.extend(["--agent-socket", self.config.harness_agent_socket])

        if self.config.harness_soft_limit:
            cmd.extend(["--soft-limit", str(self.config.harness_soft_limit)])

        if self.config.harness_controller_mode and self.config.harness_controller_mode != "standard":
            cmd.extend(["--controller-mode", self.config.harness_controller_mode])

        if self.config.harness_assessor_backend:
            cmd.extend(["--assessor-backend", self.config.harness_assessor_backend])
        if self.config.harness_assessor_model:
            cmd.extend(["--assessor-model", self.config.harness_assessor_model])
        if self.config.harness_assessor_base_url:
            cmd.extend(["--assessor-base-url", self.config.harness_assessor_base_url])
        if self.config.harness_assessor_api_key:
            cmd.extend(["--assessor-api-key", self.config.harness_assessor_api_key])
        if self.config.harness_assessor_effort:
            cmd.extend(["--assessor-effort", self.config.harness_assessor_effort])

        if self.config.harness_codex_assessor:
            cmd.append("--codex-assessor")
            if self.config.harness_codex_assessor_binary:
                cmd.extend(["--codex-assessor-binary", self.config.harness_codex_assessor_binary])
            if self.config.harness_codex_assessor_model:
                cmd.extend(["--codex-assessor-model", self.config.harness_codex_assessor_model])
            if self.config.harness_codex_assessor_effort:
                cmd.extend(["--codex-assessor-effort", self.config.harness_codex_assessor_effort])

        if self.config.cc_binary and self.config.harness_backend == "claude-code":
            cmd.extend(["--cc-binary", self.config.cc_binary])

        def _mask_cmd(c: list[str]) -> list[str]:
            masked = list(c)
            for i, arg in enumerate(masked):
                if i > 0 and "=" not in masked[i - 1] and any(k in masked[i - 1].lower() for k in ("key", "token")):
                    masked[i] = f"***{arg[-4:]}" if len(arg) > 4 else "***"
                elif "=" in arg and any(k in arg.split("=", 1)[0].lower() for k in ("key", "token")):
                    flag, val = arg.split("=", 1)
                    masked[i] = f"{flag}=***{val[-4:]}" if len(val) > 4 else f"{flag}=***"
            return masked

        logger.info("Mesh harness subprocess cmd: %s", " ".join(_mask_cmd(cmd)))

        harness_env = _build_subprocess_env()
        if self.config.node_id:
            harness_env["MESH_NODE_ID"] = self.config.node_id

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=harness_env,
            start_new_session=True,
        )

        _stderr_lines: list[str] = []

        async def _forward_stderr():
            """Forward harness subprocess stderr to parent logger."""
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                if text:
                    _stderr_lines.append(text)
                    logger.info("harness[%d]: %s", proc.pid, text)

        if proc.stdin:
            try:
                proc.stdin.write(prompt.encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            proc.stdin.close()

        stderr_task = asyncio.create_task(_forward_stderr())

        harness_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
        }
        final_text = ""
        text_blocks: list[str] = []
        buffer = b""

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        proc.stdout.read(1024 * 1024),
                        timeout=self.config.cc_idle_timeout or 600,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Mesh harness subprocess timeout — no output")
                    raise RuntimeError("Mesh harness subprocess timeout: no output")

                if not chunk:
                    if buffer.strip():
                        try:
                            event = json.loads(buffer.decode())
                            final_text = self._process_harness_event(
                                event, harness_usage, text_blocks, final_text, callback,
                            )
                        except json.JSONDecodeError:
                            pass
                    break

                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode().strip()
                    if not line_str:
                        continue
                    try:
                        event = json.loads(line_str)
                        final_text = self._process_harness_event(
                            event, harness_usage, text_blocks, final_text, callback,
                        )
                    except json.JSONDecodeError:
                        continue

            await proc.wait()
            await stderr_task

            # Always record usage so callers see what was consumed even on crash.
            self._last_usage = {
                **harness_usage,
                "total_tokens": harness_usage["input_tokens"] + harness_usage["output_tokens"],
                "backend": "mesh-harness",
                "model": model,
            }

            if proc.returncode != 0:
                stderr_str = "\n".join(_stderr_lines)
                # Preserve ALL partial assistant text, not just the most recent
                # `final_text` chunk — the harness may have emitted reasoning across
                # multiple text blocks before crashing.
                partial_text = "\n\n".join(text_blocks)
                logger.warning(
                    f"Mesh harness exited {proc.returncode}: "
                    f"{stderr_str.strip()[:500]} "
                    f"(partial_text={len(partial_text)} chars)"
                )
                # Persist the full crash record before raising; never let a
                # log-write failure mask the underlying crash.
                try:
                    log_path = _persist_harness_crash_log(
                        agent_label=_harness_agent_label(self.config),
                        returncode=proc.returncode,
                        stderr=stderr_str,
                        partial_text=partial_text,
                        model=model,
                        prompt_len=len(prompt),
                        usage=dict(self._last_usage),
                    )
                    if log_path:
                        logger.warning(f"Mesh harness crash log written: {log_path}")
                except Exception as e:
                    logger.warning(f"Mesh harness crash log persist failed: {e}")
                raise MeshHarnessCrashError(
                    returncode=proc.returncode,
                    partial_text=partial_text,
                    stderr=stderr_str,
                )

            if not final_text and text_blocks:
                final_text = "\n\n".join(text_blocks)

            logger.debug(f"Mesh harness response: len={len(final_text)}")
            return final_text

        finally:
            stderr_task.cancel()
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5.0)
                except (asyncio.TimeoutError, ProcessLookupError):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass

    def _process_harness_event(
        self,
        event: dict,
        usage: dict,
        text_blocks: list[str],
        final_text: str,
        callback: LLMStreamCallback | None,
    ) -> str:
        """Process a single mesh harness JSONL event."""
        event_type = event.get("type")
        data = event.get("data", {})

        if event_type == "thread.finished":
            final_text = data.get("final_text", final_text)
            thread_usage = data.get("usage", {})
            if isinstance(thread_usage, dict):
                usage["input_tokens"] = thread_usage.get("input_tokens", usage["input_tokens"])
                usage["output_tokens"] = thread_usage.get("output_tokens", usage["output_tokens"])
                usage["cache_creation_tokens"] = thread_usage.get("cache_creation_tokens", usage.get("cache_creation_tokens", 0))
                usage["cache_read_tokens"] = thread_usage.get("cache_read_tokens", usage.get("cache_read_tokens", 0))

        elif event_type == "usage":
            if isinstance(data, dict):
                usage["input_tokens"] += data.get("input_tokens", 0)
                usage["output_tokens"] += data.get("output_tokens", 0)
                usage["cache_creation_tokens"] += data.get("cache_creation_tokens", 0)
                usage["cache_read_tokens"] += data.get("cache_read_tokens", 0)

        elif event_type == "assistant.message":
            text = data.get("text", "")
            if text.strip():
                text_blocks.append(text.strip())

        elif event_type == "tool_call":
            logger.debug("Harness tool_call: %s", data.get("name", "unknown"))
            if callback and hasattr(callback, "on_cc_tool_event"):
                callback.on_cc_tool_event(CCToolEvent(
                    event_type="tool_call",
                    call_id=data.get("call_id", f"harness-{uuid.uuid4().hex[:8]}"),
                    tool_name=f"harness:{data.get('name', 'unknown')}",
                    data=data.get("arguments", {}),
                ))

        elif event_type == "tool_result":
            if callback and hasattr(callback, "on_cc_tool_event"):
                result_str = data.get("result", "")
                if not isinstance(result_str, str):
                    result_str = str(result_str)
                callback.on_cc_tool_event(CCToolEvent(
                    event_type="tool_result",
                    call_id=data.get("call_id", f"harness-{uuid.uuid4().hex[:8]}"),
                    tool_name=f"harness:{data.get('name', 'unknown')}",
                    data=result_str,
                ))

        elif event_type and event_type.startswith("assessor."):
            logger.info("Harness %s: %s", event_type, json.dumps(data, default=str)[:1000])
            if callback and hasattr(callback, "on_cc_tool_event"):
                call_id = f"harness-assessor-{uuid.uuid4().hex[:8]}"
                summary = ""
                if event_type == "assessor.triage":
                    summary = f"triage: decision={data.get('decision')}, needs_plan={data.get('needs_plan')}"
                elif event_type == "assessor.phase_start":
                    summary = f"phase {data.get('phase_id')} ({data.get('phase_type')}) started"
                elif event_type == "assessor.phase_complete":
                    summary = f"phase {data.get('phase_id')} ({data.get('phase_type')}) complete"
                elif event_type == "assessor.assessment":
                    summary = (
                        f"phase {data.get('phase_id')} assessment: "
                        f"decision={data.get('decision')}, "
                        f"reasoning={str(data.get('reasoning', ''))[:200]}"
                    )
                callback.on_cc_tool_event(CCToolEvent(
                    event_type="tool_call",
                    call_id=call_id,
                    tool_name=f"harness:{event_type}",
                    data=data,
                ))
                callback.on_cc_tool_event(CCToolEvent(
                    event_type="tool_result",
                    call_id=call_id,
                    tool_name=f"harness:{event_type}",
                    data=summary,
                ))

        return final_text

    def _parse_claude_code_output(self, output: str) -> str:
        """Parse Claude Code stream-json output and extract final response text."""
        final_result = ""

        for line in output.strip().split("\n"):
            if not line.strip():
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")

            if event_type == "result":
                # Final result event contains the complete response
                final_result = event.get("result", "")

        return final_result

    async def complete_with_history(
        self,
        history: list[HistoryMessage],
        node_id: str,
        system_prompt: str = "",
        tool_prompt: str = "",
        **kwargs,
    ) -> str:
        """
        Convenience method: format history and complete in one call.

        Args:
            history: Conversation history
            node_id: This agent's ID
            system_prompt: Custom system prompt
            tool_prompt: Generated tool usage instructions
            **kwargs: Passed to complete()

        Returns:
            The LLM's response text
        """
        prompt = self.format_history_xml(history, node_id, system_prompt, tool_prompt)
        return await self.complete(prompt, **kwargs)

    async def complete_with_tools(
        self,
        history: list[HistoryMessage],
        node_id: str,
        system_prompt: str,
        tool_registry: "ToolRegistry",
        tool_names: list[str] | None,
        model: str | None = None,
        callback: LLMStreamCallback | None = None,
        instructions: str = "",
        trigger_msg: "Any | None" = None,
        mcp_config: str | None = None,
        cc_system_prompt: str | None = None,
        cc_user_prompt: str | None = None,
        cc_watchdog: Any = None,
    ) -> tuple[str, list["ToolCall"]]:
        """
        Complete with tool support.

        For OpenAI backend: uses native function calling
        For other backends: uses XML tools in prompt

        Args:
            history: Conversation history
            node_id: This agent's ID
            system_prompt: Custom system prompt
            tool_registry: The tool registry to get tools from
            tool_names: List of enabled tool names (None for all)
            model: Optional model override
            callback: Optional callback for streaming events (CC tool calls, etc.)
            instructions: Optional task-specific instructions to replace default "Respond to latest message"

        Returns:
            Tuple of (response_text, tool_calls) where tool_calls is empty if none.
        """
        from .tools import ToolCall, parse_tool_calls, has_tool_call

        model = model or self.config.model
        backend = self.config.backend

        # Briefing mode: caller provides both system prompt and user prompt directly.
        # Skip format_history_xml() entirely — the worker sees a clean CC experience.
        if cc_system_prompt and cc_user_prompt and backend in ("claude-code", "zai"):
            response = await self.complete(
                cc_user_prompt, model=model, callback=callback,
                mcp_config=mcp_config,
                cc_system_prompt=cc_system_prompt,
                cc_watchdog=cc_watchdog,
            )
            return response, []

        # For OpenAI backend with tools, use native function calling
        if backend == "openai" and tool_names:
            # Build prompt WITH common guidance but OpenAI syntax (tools are passed via API)
            openai_tool_prompt = tool_registry.format_tools_prompt(tool_names, backend="openai")
            prompt = self.format_history_xml(history, node_id, system_prompt, tool_prompt=openai_tool_prompt, instructions=instructions, trigger_msg=trigger_msg)
            prompt = sanitize_prompt(prompt)

            # Store prompt for native multi-turn reasoning passback (DeepSeek)
            self._last_prompt = prompt

            # Extract images from history for vision models
            all_images = self._extract_images_from_history(history)

            # Call with tools - use Responses API for reasoning models, Chat Completions for others
            max_tokens = self.config.max_tokens
            temperature = self.config.temperature

            if self.config.should_use_responses_api(model):
                # Responses API uses flattened tool format
                openai_tools = tool_registry.get_openai_responses_tools(tool_names)
                result = await self._complete_openai_responses(prompt, model, max_tokens, tools=openai_tools, images=all_images)
            else:
                # Chat Completions API uses nested function format
                openai_tools = tool_registry.get_openai_tools(tool_names)
                result = await self._complete_openai(prompt, model, max_tokens, temperature, tools=openai_tools, images=all_images)

            # Parse result
            self._log_usage()
            if isinstance(result, tuple):
                content, openai_tool_calls = result
                # Convert OpenAI tool calls to our ToolCall format
                tool_calls = self._convert_openai_tool_calls(openai_tool_calls)
                return content, tool_calls
            else:
                return result, []

        # For Anthropic backend with tools, use native tool calling
        if backend == "anthropic" and tool_names:
            # Build prompt WITH common guidance (tools are passed via API)
            anthropic_tool_prompt = tool_registry.format_tools_prompt(tool_names, backend="openai")  # Same guidance format
            prompt = self.format_history_xml(history, node_id, system_prompt, tool_prompt=anthropic_tool_prompt, instructions=instructions, trigger_msg=trigger_msg)
            prompt = sanitize_prompt(prompt)

            # Extract images from history for vision models
            all_images = self._extract_images_from_history(history)

            # Get Anthropic-formatted tools
            anthropic_tools = tool_registry.get_anthropic_tools(tool_names)
            max_tokens = self.config.max_tokens
            temperature = self.config.temperature

            result = await self._complete_anthropic(prompt, model, max_tokens, temperature, tools=anthropic_tools, images=all_images)

            # Parse result
            self._log_usage()
            if isinstance(result, tuple):
                content, anthropic_tool_calls = result
                # Convert Anthropic tool calls to our ToolCall format (same format as OpenAI)
                tool_calls = self._convert_openai_tool_calls(anthropic_tool_calls)
                return content, tool_calls
            else:
                return result, []

        # CC + MCP: tools are discovered natively via MCP sidecar — no XML tool prompt
        # mcp_config being non-None is the authoritative signal (caller gates on cc_use_mcp)
        if backend in ("claude-code", "zai") and mcp_config:
            prompt = self.format_history_xml(
                history, node_id, system_prompt,
                tool_prompt="",  # no XML tools block
                instructions=instructions,
                trigger_msg=trigger_msg,
            )
            response = await self.complete(
                prompt, model=model, callback=callback,
                mcp_config=mcp_config,
                cc_watchdog=cc_watchdog,
            )
            # CC handles tools internally via MCP — no XML tool calls to parse
            return response, []

        # XML fallback for backends without native function calling (codex, mesh-harness, cc/zai without MCP).
        # Being replaced by shell-based mesh-tool CLI.
        import warnings
        warnings.warn(
            f"XML tool path invoked for backend={backend} — migrate to shell-based mesh-tool CLI",
            DeprecationWarning, stacklevel=2,
        )
        tool_prompt = tool_registry.format_tools_prompt(tool_names) if tool_names else ""
        prompt = self.format_history_xml(history, node_id, system_prompt, tool_prompt, instructions=instructions, trigger_msg=trigger_msg)
        response = await self.complete(prompt, model=model, callback=callback, cc_watchdog=cc_watchdog)

        # Parse XML tool calls
        if tool_names and has_tool_call(response):
            tool_calls = parse_tool_calls(response)
            return response, tool_calls

        return response, []

    async def complete_multi_turn(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        parallel_tool_calls: bool = True,
    ) -> tuple[str, list["ToolCall"], dict]:
        """OpenAI Chat Completions call with native multi-turn messages.

        Unlike complete_with_tools (which XML-serializes history into a
        single user message), this accepts pre-built messages in OpenAI
        format and passes them directly to the API.

        Returns:
            (content, tool_calls, usage) where tool_calls may be empty.
        """
        from .tools import ToolCall

        model = model or self.config.model
        max_tokens = self.config.max_tokens
        temperature = self.config.temperature
        client = self._ensure_client()

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        thinking_enabled = bool(self.config.thinking_budget and self.config.thinking_budget > 0)

        messages = _sanitize_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if not thinking_enabled:
            payload["temperature"] = temperature

        if tools:
            payload["tools"] = tools
            if parallel_tool_calls:
                payload["parallel_tool_calls"] = True

        reasoning_effort = self.config.get_effective_reasoning_effort(model)
        if reasoning_effort and reasoning_effort != "none":
            payload["reasoning_effort"] = reasoning_effort
            if "deepseek" in (self.config.base_url or "").lower():
                payload["thinking"] = {"type": "enabled"}

        if thinking_enabled:
            payload["enable_thinking"] = True
            payload["thinking_budget"] = self.config.thinking_budget

        logger.debug(
            "LLM multi-turn request: model=%s url=%s messages=%d tools=%d thinking=%s",
            model, url, len(messages), len(tools) if tools else 0,
            self.config.thinking_budget or "off",
        )

        async def _make_request() -> tuple[str, list[ToolCall], dict]:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 400 and "reasoning_effort" in payload:
                error_text = response.text
                effort_val = payload.get("reasoning_effort")
                if effort_val == "xhigh" and "high" in error_text:
                    payload["reasoning_effort"] = "high"
                    response = await client.post(url, headers=headers, json=payload)
                elif "non-reasoning model" in error_text or "does not support reasoning" in error_text:
                    payload.pop("reasoning_effort")
                    response = await client.post(url, headers=headers, json=payload)

            if 400 <= response.status_code < 500:
                logger.warning(
                    "Multi-turn %d from %s (model=%s): body[:1000]=%s",
                    response.status_code, url, model, response.text[:1000],
                )

            response.raise_for_status()
            data = response.json()

            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                raise MalformedResponseError(
                    f"Multi-turn response missing 'choices' (model={model}): "
                    f"body[:200]={str(data)[:200]!r}"
                )
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise MalformedResponseError(
                    f"Multi-turn response choices[0] missing 'message' (model={model})"
                )

            content = message.get("content") or ""
            self._last_raw_message = message
            reasoning_content = message.get("reasoning_content") or ""
            self._last_reasoning_content = reasoning_content.strip() if reasoning_content else None
            if not content.strip() and reasoning_content.strip():
                content = reasoning_content.strip()

            usage = data.get("usage", {})
            ct_details = usage.get("completion_tokens_details") or {}
            pt_details = usage.get("prompt_tokens_details") or {}
            cached_tokens = 0
            if isinstance(pt_details, dict):
                cached_tokens = pt_details.get("cached_tokens", 0) or 0
            if not cached_tokens:
                cached_tokens = usage.get("prompt_cache_hit_tokens", 0) or 0
            self._last_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_tokens": 0,
                "cache_read_tokens": cached_tokens,
                "reasoning_tokens": ct_details.get("reasoning_tokens", 0) if isinstance(ct_details, dict) else 0,
                "total_tokens": usage.get("total_tokens", 0) or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
                "backend": "openai",
                "model": model,
            }

            raw_tool_calls = message.get("tool_calls") or []
            tool_calls = self._convert_openai_tool_calls(raw_tool_calls)
            return content, tool_calls, dict(self._last_usage)

        return await self._retry_with_backoff(_make_request, f"Multi-turn request ({model})")

    def _extract_images_from_history(self, history: list[HistoryMessage]) -> list[ImageAttachment] | None:
        """Extract all images from history messages.

        Returns None if no images, or a list of ImageAttachment objects.
        This collects images from all messages to include in a single vision API call.
        """
        all_images = []
        for msg in history:
            if msg.images:
                all_images.extend(msg.images)
        return all_images if all_images else None

    def _convert_openai_tool_calls(self, openai_tool_calls: list[dict]) -> list["ToolCall"]:
        """
        Convert OpenAI tool calls to our ToolCall format.

        OpenAI format:
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\": \"Austin\"}"
            }
        }

        Our format:
        ToolCall(name="get_weather", arguments={"location": "Austin"}, raw_xml="")
        """
        from .tools import ToolCall

        result = []
        for tc in openai_tool_calls:
            if tc.get("type") != "function":
                continue

            func = tc.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "{}")

            # Parse arguments JSON
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {}

            # Generate synthetic XML for compatibility with existing code
            # This is used for logging and debugging
            raw_xml = f'<mesh_call name="{name}">'
            for k, v in arguments.items():
                raw_xml += f"<{k}>{json.dumps(v) if not isinstance(v, str) else v}</{k}>"
            raw_xml += "</mesh_call>"

            result.append(ToolCall(name=name, arguments=arguments, raw_xml=raw_xml, call_id=tc.get("id", "")))

        return result
