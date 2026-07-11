# SPDX-License-Identifier: Apache-2.0
"""
Configuration loading for the mesh.

Loads settings from YAML config file with sensible defaults.
Designed with future authentication in mind (placeholder fields).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class EffortPreset(str, Enum):
    """
    Effort presets control how thorough the v0.2 controller is.

    Higher effort = lower thresholds = more thorough work.
    Lower effort = higher thresholds = faster, less thorough.
    """
    HIGH = "high"      # Thorough: lower thresholds, more phases
    MEDIUM = "medium"  # Balanced: default thresholds
    LOW = "low"        # Quick: higher thresholds, fewer phases


# Threshold lookup table for effort presets
# Keys: (preset, threshold_name) -> value
EFFORT_THRESHOLDS: dict[tuple[EffortPreset, str], float] = {
    # HIGH effort: more info gathering, more phases
    (EffortPreset.HIGH, "info"): 0.2,           # Low bar for info gathering
    (EffortPreset.HIGH, "complexity_low"): 0.2,  # Few tasks go to fast path
    (EffortPreset.HIGH, "complexity_high"): 0.5, # Many tasks get full treatment
    (EffortPreset.HIGH, "plan_quality"): 0.7,    # Demand good plans

    # MEDIUM effort: balanced defaults
    (EffortPreset.MEDIUM, "info"): 0.3,
    (EffortPreset.MEDIUM, "complexity_low"): 0.3,
    (EffortPreset.MEDIUM, "complexity_high"): 0.7,
    (EffortPreset.MEDIUM, "plan_quality"): 0.6,

    # LOW effort: quick and direct
    (EffortPreset.LOW, "info"): 0.5,             # High bar for info gathering
    (EffortPreset.LOW, "complexity_low"): 0.4,   # More tasks go fast path
    (EffortPreset.LOW, "complexity_high"): 0.8,  # Few tasks get full treatment
    (EffortPreset.LOW, "plan_quality"): 0.5,     # Accept lower quality plans
}


def get_effort_threshold(preset: EffortPreset, threshold_name: str) -> float:
    """
    Get a threshold value for a given effort preset.

    Args:
        preset: The effort preset (HIGH, MEDIUM, LOW)
        threshold_name: One of "info", "complexity_low", "complexity_high", "plan_quality"

    Returns:
        The threshold value (0.0 to 1.0)

    Raises:
        KeyError: If threshold_name is not recognized
    """
    return EFFORT_THRESHOLDS[(preset, threshold_name)]


# Default prompts directory relative to this file
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt_file(filename: str, prompts_dir: Path | None = None) -> str:
    """
    Load a prompt from a file in the prompts directory.

    If the file is an agent prompt (not a shared include file), this function
    automatically appends the channel_policy.md content so agents understand
    how to behave in channels.

    Args:
        filename: Filename (e.g., "researcher.md") or relative path
        prompts_dir: Override prompts directory (defaults to mesh/prompts/)

    Returns:
        The prompt content as a string, or empty string if file not found.
    """
    if prompts_dir is None:
        prompts_dir = PROMPTS_DIR

    prompt_path = prompts_dir / filename
    if not prompt_path.exists():
        return ""

    content = prompt_path.read_text().strip()

    # Auto-include channel_policy.md for agent prompts
    # Skip for shared/include files like channel_policy.md, tool_instructions.md
    shared_files = {"channel_policy.md", "tool_instructions.md", "memory.md", "mesh_tools.md"}
    if filename not in shared_files:
        channel_policy_path = prompts_dir / "channel_policy.md"
        if channel_policy_path.exists():
            channel_policy = channel_policy_path.read_text().strip()
            content = content + "\n\n" + channel_policy

        memory_path = prompts_dir / "memory.md"
        if memory_path.exists():
            memory = memory_path.read_text().strip()
            content = content + "\n\n" + memory

        mesh_tools_path = prompts_dir / "mesh_tools.md"
        if mesh_tools_path.exists():
            mesh_tools = mesh_tools_path.read_text().strip()
            content = content + "\n\n" + mesh_tools

    return content


@dataclass
class RelevanceRouterConfig:
    """
    Configuration for the LLM-based relevance router.

    Used to filter channel messages - decides whether an agent should
    respond based on relevance scoring.
    """
    # Score threshold for processing (0.0-1.0)
    # Messages scoring below this are ignored
    threshold: float = 0.7

    # Fast-path bypasses (skip LLM call)
    bypass_direct: bool = True     # Direct messages always process
    bypass_mentions: bool = False  # Nickname mentions bypass (when True, like current behavior)

    # LLM settings for relevance scoring
    model: str = "gpt-4o-mini"     # Small/fast model for scoring
    backend: str = "openai"


@dataclass
class RouterConfig:
    """Configuration for the router/broker."""
    host: str = "127.0.0.1"
    port: int = 7700
    storage_path: str = "~/log/chats/mesh-storage/messages.db"

    # WebSocket settings (for browser/mobile clients)
    ws_enabled: bool = True
    ws_port: int = 8080

    # Authentication settings
    auth_enabled: bool = False
    auth_token: str | None = None  # Global token (all nodes use this)
    auth_tokens: dict[str, str] = field(default_factory=dict)  # Per-node tokens
    auth_mode: str = "global"  # "global" (single token) or "per_user" (user table)

    # Attachment settings (download URLs are short-lived bearer URLs)
    attachments_enabled: bool = True
    attachments_dir: str = "~/.mesh/attachments"
    attachments_max_file_bytes: int = 50 * 1024 * 1024
    attachments_per_owner_quota_bytes: int = 500 * 1024 * 1024
    attachments_signing_secret: str | None = None
    attachments_url_ttl_secs: int = 600

    # FCM (Firebase Cloud Messaging) settings
    fcm_enabled: bool = False
    fcm_credentials_file: str | None = None  # Path to service account JSON

    def __post_init__(self):
        from .paths import resolve_path
        self.storage_path = resolve_path(self.storage_path)
        # Expand environment variable references in auth_token
        if self.auth_token and self.auth_token.startswith("${") and self.auth_token.endswith("}"):
            env_var = self.auth_token[2:-1]
            self.auth_token = os.environ.get(env_var, "")
        self.attachments_dir = resolve_path(self.attachments_dir)
        if (
            self.attachments_signing_secret
            and self.attachments_signing_secret.startswith("${")
            and self.attachments_signing_secret.endswith("}")
        ):
            env_var = self.attachments_signing_secret[2:-1]
            self.attachments_signing_secret = os.environ.get(env_var, "")
        # Expand path for FCM credentials
        if self.fcm_credentials_file:
            self.fcm_credentials_file = resolve_path(self.fcm_credentials_file)


@dataclass
class LLMBackendConfig:
    """
    Configuration for an LLM backend.

    Supported backend types:
    - "openai": OpenAI-compatible APIs (default)
    - "anthropic": Anthropic Claude via native API
    - "claude-code": Claude Code subprocess (claude -p)
    - "zai": Z.AI via Claude Code with Z.AI proxy
    - "codex": Codex CLI subprocess (codex exec --json)
    - "mesh-harness": Standalone harness subprocess (python -m mesh.harness exec)
    """
    # Backend type: openai, anthropic, claude-code, zai, codex, mesh-harness
    backend_type: str = "openai"

    # API key (can use env var reference like ${OPENAI_API_KEY})
    api_key: str = ""

    # OpenAI-compatible settings
    base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7

    # Claude Code / Z.AI settings
    cc_allowed_tools: list[str] = field(default_factory=lambda: ["Read", "Edit", "Bash"])
    cc_fallback_homes: list[str] = field(default_factory=list)  # Fallback HOME dirs for multi-account CC

    # Thinking/Reasoning model settings
    # For OpenAI (o3, o4, gpt-5): reasoning_effort
    # For Google Gemini 3.x: thinking_level
    # For Google Gemini 2.5: thinking_budget
    # For Anthropic: anthropic_thinking_budget
    reasoning_effort: str | None = None      # "none", "low", "medium", "high"
    thinking_level: str | None = None        # "none", "low", "medium", "high" (Gemini 3)
    thinking_budget: int | None = None       # 0-24576 or -1 for dynamic (Gemini 2.5)
    anthropic_thinking_budget: int | None = None  # budget_tokens for Anthropic extended thinking
    include_thoughts: bool = False           # Include thinking content in response
    auto_detect_reasoning: bool = True       # Auto-detect reasoning models

    # Cookie-based auth (e.g., TAMU Cloudflare Access)
    cookie_source: str = ""  # "tamu" → inject CF cookies from ~/.mesh/tamu_cookies.json

    # Claude Code subprocess environment overrides
    # Merged into CC subprocess env (e.g., ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY
    # to point CC at a shim/proxy). Generalizes the old ZAI-specific env pattern.
    cc_env: dict[str, str] | None = None

    # Explicit thinking mode override for Claude Code backend.
    # - None: let CC auto-decide based on model name (openai/* → thinking, else no)
    # - True: force thinking ON (model name prefixed with "openai/" if needed)
    # - False: force thinking OFF (model name stripped of "openai/" prefix if present)
    # Note: CC has no --thinking flag; thinking is triggered by model name prefix.
    # The shim's wildcard route handles any model name sent by CC.
    thinking: bool | None = None

    # CC --effort flag.  Controls reasoning depth, tool call volume, and
    # response thoroughness.  Valid values depend on CC version:
    #   v2.1.77: low, medium, high
    #   v2.1.111+: low, medium, high, xhigh, max
    # Empty string = let CC use its default.
    cc_effort: str = "high"

    # Path to the Claude Code binary.  Allows pinning to a known-good version
    # instead of whatever `claude` resolves to in PATH.
    # Empty string = auto-detect via shutil.which("claude").
    cc_binary: str = ""

    # MCP integration: when True, CC workers discover mesh tools via MCP sidecar
    # instead of XML <mesh_call> syntax.  Requires CC >= 2.1.114.
    cc_use_mcp: bool = False

    # Worker briefing: when True, router generates a condensed briefing for CC
    # workers instead of passing the full conversation history.  The briefing
    # lives in --system-prompt (durable under CC compaction).
    cc_worker_briefing: bool = False

    # Codex CLI settings (backend_type: codex)
    codex_binary: str = ""  # Path to codex binary; empty = shutil.which("codex")

    # Codex idle timeout in seconds.  Codex can spend long stretches in internal
    # turns without writing to stdout, so idle-timeout watchdogs are usually wrong.
    # 0 = disabled (recommended).  Use only as a last-resort runaway guard.
    codex_subprocess_idle_timeout: int = 0

    # Extra CLI args appended to `codex exec` (e.g. ["--sandbox", "read-only",
    # "--disable", "shell_tool"]).  When this list contains `--sandbox`, the
    # default `--dangerously-bypass-approvals-and-sandbox` flag is dropped.
    codex_extra_args: list[str] = field(default_factory=list)

    # Mesh harness settings (backend_type: mesh-harness)
    harness_python: str = ""  # Python binary; empty = sys.executable
    harness_backend: str = "anthropic"  # Sub-backend for harness LLM calls
    harness_base_url: str = ""  # API base URL for harness sub-backend
    harness_api_key: str = ""  # API key for harness sub-backend (env var ref OK)
    harness_toolset: str = "legacy"  # "harness" (4-tool) or "legacy" (full mesh tools)
    harness_tools: str = ""  # Comma-separated tool names; overrides harness_toolset
    harness_system_prompt_file: str = ""  # System prompt file path
    harness_soft_limit: int = 0  # Token soft limit for harness context (0 = use harness default)
    harness_controller_mode: str = "standard"  # "standard" | "plan_and_execute" | "decompose"
    harness_compaction_threshold_fraction: float = 0.40
    harness_max_phases: int = 15

    # Assessor LLM settings (for plan-and-execute mode)
    harness_assessor_backend: str = ""  # e.g., "openai"
    harness_assessor_model: str = ""  # e.g., "deepseek-v4-pro"
    harness_assessor_base_url: str = ""  # e.g., "https://api.deepseek.com/v1"
    harness_assessor_api_key: str = ""  # API key for assessor (env var ref OK)
    harness_assessor_effort: str = ""  # Reasoning effort for assessor

    # Codex assessor settings (subprocess-based controller using codex exec)
    harness_codex_assessor: bool = False
    harness_codex_assessor_binary: str = ""  # Empty = shutil.which("codex")
    harness_codex_assessor_model: str = "o3"
    harness_codex_assessor_effort: str = "high"

    def __post_init__(self):
        # Expand environment variable references in api_key
        if self.api_key.startswith("${") and self.api_key.endswith("}"):
            env_var = self.api_key[2:-1]
            self.api_key = os.environ.get(env_var, "")
        if self.harness_api_key.startswith("${") and self.harness_api_key.endswith("}"):
            env_var = self.harness_api_key[2:-1]
            self.harness_api_key = os.environ.get(env_var, "")
        if self.harness_assessor_api_key.startswith("${") and self.harness_assessor_api_key.endswith("}"):
            env_var = self.harness_assessor_api_key[2:-1]
            self.harness_assessor_api_key = os.environ.get(env_var, "")


@dataclass
class ControllerConfig:
    """
    Configuration for message routing and task management controllers.

    Controllers sit between incoming messages and the LLM, enabling:
    - Task tracking and routing
    - Workflow management (phases, plans)
    - Edit proposal and approval flows

    Modes:
    - "passthrough": Default, preserves existing behavior (direct LLM pass-through)
    - "task-fsm-v0": Rule-based router + hardcoded phase FSM
    - "task-fsm-v1": Future learned components (RL router)
    """
    # Controller mode
    mode: str = "passthrough"  # "passthrough" | "task-fsm-v0" | "task-fsm-v1"

    # Persistence paths (expanded with os.path.expanduser)
    tasks_path: str = "~/log/assistant/tasks.json"
    config_path: str = "~/log/assistant/config.json"

    # Router LLM settings (for task-fsm modes)
    router_model: str = "gpt-4o-mini"
    router_backend: str = "openai"

    # Confidence threshold for routing decisions
    # Below this threshold, ask for clarification instead of routing
    confidence_threshold: float = 0.4

    # Edit approval settings
    # When True, file writes require user approval via /approve
    # When False, file writes are auto-approved (no permission prompts)
    require_edit_approval: bool = True

    def __post_init__(self):
        # Expand user paths (uses real home, not CC synthetic home)
        from .paths import resolve_path
        self.tasks_path = resolve_path(self.tasks_path)
        self.config_path = resolve_path(self.config_path)


@dataclass
class ControllerConfigV02:
    """
    Configuration for the v0.2 stateless phase-flow controller.

    The v0.2 controller uses LLM-scored adaptive phases instead of
    hard-coded transitions. It is STATELESS between messages - no
    task persistence, no RouterLLM.

    Modes:
    - "passthrough": Direct LLM pass-through (no controller)
    - "phase-flow-v02": LLM-scored adaptive phase flow
    """
    # Controller mode
    mode: str = "passthrough"  # "passthrough" | "phase-flow-v02"

    # Effort preset (controls all thresholds)
    # Use EffortPreset enum values: "low", "medium", "high"
    effort: str = "medium"

    # Individual threshold overrides (if set, override effort preset)
    # All values 0.0 to 1.0
    info_threshold: float | None = None       # Override for info gathering
    complexity_low: float | None = None       # Override for LOW complexity cutoff
    complexity_high: float | None = None      # Override for HIGH complexity cutoff
    plan_quality: float | None = None         # Override for plan quality threshold

    # Max iterations before forcing forward
    max_info_iterations: int = 3
    max_plan_iterations: int = 3

    # Metrics tracking
    enable_metrics: bool = False              # Track LLM calls, tokens, timing

    # Streaming/observability
    stream_phase_updates: bool = True         # Send phase transition messages to user

    def get_effort_preset(self) -> EffortPreset:
        """Get the effort preset as an enum."""
        return EffortPreset(self.effort)

    def get_threshold(self, name: str) -> float:
        """
        Get a threshold value, respecting individual overrides.

        Args:
            name: One of "info", "complexity_low", "complexity_high", "plan_quality"

        Returns:
            The threshold value (individual override or from effort preset)
        """
        # Check for individual override
        override_map = {
            "info": self.info_threshold,
            "complexity_low": self.complexity_low,
            "complexity_high": self.complexity_high,
            "plan_quality": self.plan_quality,
        }
        override = override_map.get(name)
        if override is not None:
            return override

        # Fall back to effort preset
        return get_effort_threshold(self.get_effort_preset(), name)


@dataclass
class CCSessionConfig:
    """Configuration for CC-session executor mode (per-agent)."""
    # System prompt composition — which blocks to include
    system_prompt_includes: list[str] = field(default_factory=lambda: [
        "identity", "personality", "memories", "rolling_summary",
        "retrieved_context", "mesh_protocol", "communication",
    ])

    # Memory injection strategy: how/when memories are refreshed in the system prompt
    memory_refresh: str = "per-turn"  # "per-turn" | "tool" | "restart" | "interval:Xh"

    # Session lifetime policy
    session_lifetime: str = "persistent"  # "persistent" | "per-topic" | "per-day"

    # Session state directory (session ID files persisted here)
    session_dir: str = "~/.mesh/sessions"

    # CC process settings
    cc_output_format: str = "stream-json"
    cc_model: str | None = None  # Override model for CC process (None = use backend default)
    cc_max_turns: int | None = None  # Max CC turns per invocation (None = unlimited)

    # System prompt token budget
    system_prompt_budget_tokens: int = 10_000

    def __post_init__(self):
        from .paths import resolve_path
        self.session_dir = resolve_path(self.session_dir)


@dataclass
class MemoryProfileConfig:
    """Profile configuration overrides for memory rendering.

    All fields are optional — None means "use the built-in default."
    Field names MUST match MemoryProfile for _build_profile() merge to work.
    """
    budget_tokens: int | None = None
    representative_pct: float | None = None
    recent_pct: float | None = None
    relevant_pct: float | None = None
    representative_full_reflections: int | None = None
    recent_full_reflections: int | None = None
    relevant_full_reflections: int | None = None
    relevant_top_traces: int | None = None
    similarity_floor: float | None = None


@dataclass
class NodeConfig:
    """Configuration for a node (user or agent)."""
    id: str                          # e.g., "user:yourname" or "agent:researcher"
    router_host: str = "127.0.0.1"
    router_port: int = 7700
    router_ws_port: int = 8765          # WebSocket port (for MCP server connections)
    storage_path: str = "~/log/chats/mesh-storage/messages.db"

    # TLS settings
    use_tls: bool = False               # Enable TLS for router connection
    tls_server_hostname: str | None = None  # Override hostname for cert verification

    # WebSocket URL for remote connections (e.g., wss://host/mesh/ws)
    ws_url: str | None = None

    # Auth settings
    auth_token: str | None = None       # Token for router authentication

    # Nickname for display and addressing (optional)
    # If not set, auto-generated or derived from id
    nickname: str | None = None

    # Agent-specific settings
    agent_type: str | None = None     # Agent type (e.g., "coder", "researcher")
    llm_model: str | None = None      # Which LLM to use (None = use backend default)
    system_prompt: str = ""           # Custom system prompt for the agent
    system_prompt_file: str | None = None  # Path to prompt file (relative to prompts/ dir)
    llm_backend: str = "default"      # Which LLM backend to use
    tools: list[str] = field(default_factory=list)  # List of enabled tool names

    # Preference extraction settings
    # Periodically extracts user patterns from history and prepends to context
    pref_message_threshold: int | None = None    # Extract every N messages (default: 50)
    pref_context_limit: int | None = None        # Max tokens from history (default: 100k)
    pref_stale_hours: int | None = None          # Re-extract if older than N hours (default: 24)
    pref_extraction_model: str | None = None     # Model for extraction (default: sonnet)
    pref_extraction_backend: str | None = None   # Backend for extraction (default: claude-code)

    # Sandbox settings (restrict file/bash access)
    sandboxed: bool = False                       # Enable bwrap sandboxing
    allowed_dirs: list[str] = field(default_factory=list)  # Writable directories
    allow_network: bool = True                    # Allow network access in sandbox

    # Controller settings (task routing and workflow management)
    # Supports both v0.1 (ControllerConfig) and v0.2 (ControllerConfigV02)
    controller: ControllerConfig | ControllerConfigV02 | None = None

    # Processing limits
    max_processing_time: float | None = None  # Wall-clock timeout in seconds for entire request

    # Working directory for file operations (cleared on reset_context)
    workdir: str | None = None  # e.g., "/tmp/evalplus" - cleared between problems

    # Router V2: Mediating router with acks and status queries
    # When enabled, replaces relevance router with a component that:
    # - Sends immediate acks for long-running requests (>10s)
    # - Handles status queries while processing
    # - Isolates worker context from router-level messages
    use_router_v2: bool = True
    use_router_v3: bool = False   # RouterV3: adds planning pipeline (subclasses V2)

    # Router V2 LLM: When enabled, router uses LLM for classification
    # - Decides needs_worker (true/false) for incoming messages
    # - Generates contextual responses instead of canned text
    # - Uses router_v2_llm_backend/model if set, else same as worker
    router_v2_llm_enabled: bool = True

    # Separate LLM backend/model for the router (classification, busy, completion).
    # When set, router uses a DIFFERENT (typically faster/cheaper) LLM than the worker,
    # avoiding concurrency issues when both need to call the LLM simultaneously.
    # If None, falls back to llm_backend/llm_model (shared with worker).
    router_v2_llm_backend: str | None = None   # e.g., "default" for gpt-4o
    router_v2_llm_model: str | None = None     # e.g., "gpt-4o-mini"

    # History management — unified fields for both router and worker.
    # When summarization is disabled, history is a simple rolling window:
    # oldest turns are dropped when the window exceeds hard_limit_tokens.
    history_summarization_enabled: bool = False       # off = rolling window only
    history_soft_limit_tokens: int = 70_000           # rolling window cap
    history_hard_limit_tokens: int = 90_000           # hard cap (drop oldest turns)
    history_window_tokens: int | None = None           # rolling window budget W (default: soft_limit // 2)
    watchdog_interval_minutes: int = 15               # worker watchdog check-in interval (0 = disabled)
    worker_context_window_tokens: int = 25_000         # token budget for worker context snapshot
    worker_in_flight_token_limit: int = 150_000       # safety valve for tool loops (independent)

    # Router history settings (deprecated — unified fields above take precedence)
    router_history_soft_limit_tokens: int = 70_000   # deprecated: use history_soft_limit_tokens
    router_history_hard_limit_tokens: int = 90_000   # deprecated: use history_hard_limit_tokens
    router_history_target_ratio: float = 0.25        # target = soft * ratio after summarization
    router_history_persist: bool = True               # persist router history to disk

    # Memory profile configuration (three-slice rendering)
    memory_profile_light: "MemoryProfileConfig | None" = None   # Simple worker
    memory_profile_deep: "MemoryProfileConfig | None" = None    # Complex worker
    # Backward compat: accept old names during transition
    memory_router_profile: "MemoryProfileConfig | None" = None  # deprecated → maps to light
    memory_worker_profile: "MemoryProfileConfig | None" = None  # deprecated → maps to deep

    # Worker synthesis settings
    synthesize_enabled: bool = True           # Enable synthesis step on worker completion
    worker_digest_max_tokens: int = 15_000    # Token cap for worker digest (persistent)
    synthesis_max_tokens: int = 150_000       # Total token cap for synthesis prompt
    # On worker completion, deliver messages the worker buffered for the
    # dispatch origin VERBATIM (concatenated into one message) instead of
    # synthesizing a description of them. Synthesis still runs when the
    # buffer holds nothing addressed to the origin. Canary: alice.
    deliver_buffered_verbatim: bool = False

    # Trace-as-history (see docs/plans/trace-as-history-2026-04-27.md)
    trace_as_history_enabled: bool = False    # OFF by default; per-agent flip for canary
    tool_result_max_lines: int = 80           # truncation cap at append time
    tool_result_max_chars: int = 6400         # fallback for unstructured payloads

    # Memory system settings
    memory_enabled: bool = False
    memory_active_size: int = 30
    # Effectively unlimited: _prune_pool() only fires above this ceiling.
    # Raised from 1000 (2026-07-06) — FIFO pruning was silently deleting
    # oldest memories on capped agents. memory_search is a linear scan;
    # fine well past 10k entries.
    memory_pool_max_entries: int = 100000
    memory_embedding_model: str = "text-embedding-3-small"
    memory_embedding_backend: str = "openai"
    memory_reflection_min_tools: int = 3
    memory_reflection_min_discussion_turns: int = 4
    memory_reflection_min_discussion_chars: int = 1500
    memory_reflection_min_brainstorm_response_chars: int = 1500
    memory_reflection_max_brainstorm_tools: int = 2
    memory_reflection_cooldown_secs: int = 300
    memory_reflection_session_gap_secs: int = 900  # 15 min gap = new session
    memory_reflection_flush_interval_tools: int = 0  # 0 = disabled; >0 = flush every N tool calls within a worker
    memory_retrieval_k: int = 5
    memory_worker_full_reflections: int = 2
    memory_router_full_reflections: int = 0
    memory_router_recent_reflections: int = 3
    memory_worker_recent_reflections: int = 2
    memory_trace_max_tokens: int = 2000
    memory_reflection_max_tokens: int = 500

    # Memory v2 settings
    memory_version: int = 1                          # 1 = current, 2 = project-oriented
    memory_recent_log_count: int = 4                 # number of recent log entries in prompt (count-based)
    memory_retrieve_budget_tokens: int = 6000        # token budget for on-demand retrieval
    memory_retrieve_max_rounds: int = 2              # max retrieval round-trips before proceeding
    memory_curation_audit_max_tool_calls: int = 10   # safety cap on tool calls during map curation
    memory_review_max_tool_calls: int = 30           # tool call budget for interactive map review

    # Memory Formation v3 settings (see docs/plans/memory-formation-v3-2026-04-27.md)
    memory_llm_backend: str = ""                     # Separate LLM backend for memory ops (formation, etc.)
    memory_formation_v3_enabled: bool = False        # OFF by default; enable per-agent for rollout
    memory_formation_token_threshold: int = 30000    # token-pressure trigger; 0 disables
    memory_formation_interval_seconds: int = 1800    # time-based trigger interval
    memory_formation_defer_tail_seconds: int = 300   # turns younger than this skipped by time-based
    memory_formation_shutdown_timeout: float = 30.0  # cap on blocking shutdown formation
    memory_v3_window_size: int = 60                  # segmenter window
    memory_v3_overlap: int = 20                      # window overlap
    memory_v3_defer_tail: int = 10                   # in-window trailing turns deferred
    memory_v3_model: str = "deepseek-v4-flash"       # segmenter model
    memory_v3_parse_failure_fallback_threshold: int = 3  # placeholder after N consecutive failures

    # Phase 1: lowered formation bar (decoupled from digest bar).
    # When True, the fold driver uses the recall-oriented formation prompt
    # and tags each minted record with digest_candidate (True/False) based
    # on the unchanged high digest significance bar. The fold injection
    # filter shows only digest_candidate=True rows to the fold.
    memory_formation_lowbar: bool = False

    # Memory retrieval redesign (see docs/plans/memory-retrieval-redesign-2026-04-27.md)
    memory_retrieval_redesign_enabled: bool = False
    memory_toc_size: int = 30
    memory_toc_ranking: str = "cosine"  # "cosine" | "flmi" | "hybrid"

    # Rev-10 standing-digest read pathway (per-agent-standing-digest spec):
    # when enabled, the published standing digest replaces the <memory_toc>
    # block in prompt composition. Alongside-deploy: default off; the old
    # TOC pathway is untouched unless the flag is set on the agent.
    standing_digest_enabled: bool = False
    standing_digest_path: str = ""  # published digest file (see fold_driver/alice)

    memory_search_mode: str = "hybrid"  # "embedding" | "lexical" | "hybrid"

    # Fold driver LLM backend (standing-digest offline fold).
    # Resolved by _resolve_fold_backend() in the fold driver.
    fold_backend: str = "deepseek-direct"

    memory_get_payload_max_chars: int = 6000
    memory_search_default_k: int = 5

    # Router V2 full mode settings
    # "full" = conversational agent with tools; "classifier" = legacy thin classifier
    router_mode: str = "classifier"
    router_max_iters: int = 10  # Max tool-loop iterations for full router
    pipeline_backend: str = "deepseek"
    pipeline_plan_path: str = ""

    # Personality seed — initial personality text, seeded into DB on first boot.
    # Once seeded, the agent can overwrite via personality_set tool.
    personality: str = ""

    # Per-agent auto-confirm: skip CONFIRM_REQUEST for these tools
    # e.g., ["gmail_send_message", "gmail_reply_to"] lets this agent send email without user approval
    auto_confirm_tools: list[str] = field(default_factory=list)

    # Channels to auto-join on startup
    channels: list[str] = field(default_factory=list)

    # Executor mode: "rolling-window" (current V2 architecture) or "cc-session" (CC-managed context)
    context_mode: str = "rolling-window"  # "rolling-window" | "cc-session"

    # CC-session configuration (only used when context_mode == "cc-session")
    cc_session: CCSessionConfig = field(default_factory=CCSessionConfig)

    # CC interactive tools: expose tmux-based Claude Code session tools to the router LLM.
    # When True, the router gets cc_start_session, cc_get_screen, cc_send_input, cc_stop_session.
    cc_interactive_tools: bool = False
    cc_interactive_binary: str = ""
    cc_interactive_model: str = ""
    cc_interactive_effort: str = ""

    # Native harness session tools: expose the mesh-harness interactive session
    # tools to the router LLM (harness_start_session, harness_send_input,
    # harness_get_status, harness_stop_session). The native equivalent of the CC
    # interactive path — a persistent harness subprocess driven over pipes, no
    # tmux scraping. harness_session_backend names the llm_backends block the
    # session runs on (e.g. mesh-harness-qwen36).
    harness_session_tools: bool = False
    harness_session_backend: str = ""

    def __post_init__(self):
        # Expand environment variable references in auth_token
        if self.auth_token and self.auth_token.startswith("${") and self.auth_token.endswith("}"):
            env_var = self.auth_token[2:-1]
            self.auth_token = os.environ.get(env_var, "")


@dataclass
class MeshConfig:
    """Top-level configuration."""
    router: RouterConfig = field(default_factory=RouterConfig)
    nodes: dict[str, NodeConfig] = field(default_factory=dict)
    llm_backends: dict[str, LLMBackendConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> MeshConfig:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeshConfig:
        """Create config from a dictionary."""
        router_data = data.get("router", {})
        router = RouterConfig(**router_data)

        # Parse LLM backends
        llm_backends = {}
        for backend_id, backend_data in data.get("llm_backends", {}).items():
            llm_backends[backend_id] = LLMBackendConfig(**backend_data)

        nodes = {}
        for node_id, node_data in data.get("nodes", {}).items():
            # Handle nodes defined with no properties (value is None)
            if node_data is None:
                node_data = {}
            node_data["id"] = node_id
            # Inherit router settings if not specified
            node_data.setdefault("router_host", router.host)
            node_data.setdefault("router_port", router.port)
            node_data.setdefault("router_ws_port", router.ws_port)
            node_data.setdefault("storage_path", router.storage_path)

            # Load system_prompt from file if system_prompt_file is specified
            # and system_prompt is not already set inline
            if node_data.get("system_prompt_file") and not node_data.get("system_prompt"):
                prompt_content = load_prompt_file(node_data["system_prompt_file"])
                if prompt_content:
                    node_data["system_prompt"] = prompt_content

            # Parse memory profile configs if present (new + old names)
            for profile_key in (
                'memory_profile_light', 'memory_profile_deep',
                'memory_router_profile', 'memory_worker_profile',
            ):
                if profile_key in node_data and isinstance(node_data[profile_key], dict):
                    node_data[profile_key] = MemoryProfileConfig(**node_data[profile_key])

            # Parse cc_session config if present
            if "cc_session" in node_data and isinstance(node_data["cc_session"], dict):
                node_data["cc_session"] = CCSessionConfig(**node_data["cc_session"])

            # Parse controller config if present
            if "controller" in node_data and node_data["controller"] is not None:
                controller_data = node_data["controller"]
                if isinstance(controller_data, dict):
                    # Determine which controller class based on mode
                    mode = controller_data.get("mode", "passthrough")
                    if mode == "phase-flow-v02":
                        node_data["controller"] = ControllerConfigV02(**controller_data)
                    else:
                        # v0.1 modes: passthrough, task-fsm-v0, task-fsm-v1
                        node_data["controller"] = ControllerConfig(**controller_data)
                # If it's already a ControllerConfig/V02, leave it as is

            nodes[node_id] = NodeConfig(**node_data)

        return cls(router=router, nodes=nodes, llm_backends=llm_backends)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        router_dict = {
            "host": self.router.host,
            "port": self.router.port,
            "storage_path": self.router.storage_path,
            "ws_enabled": self.router.ws_enabled,
            "ws_port": self.router.ws_port,
            "auth_enabled": self.router.auth_enabled,
        }
        if self.router.auth_token:
            router_dict["auth_token"] = self.router.auth_token
        if self.router.auth_tokens:
            router_dict["auth_tokens"] = self.router.auth_tokens

        return {
            "router": router_dict,
            "llm_backends": {
                backend_id: {
                    "backend_type": backend.backend_type,
                    "api_key": backend.api_key,
                    "base_url": backend.base_url,
                    "default_model": backend.default_model,
                    "max_tokens": backend.max_tokens,
                    "temperature": backend.temperature,
                    "cc_allowed_tools": backend.cc_allowed_tools,
                }
                for backend_id, backend in self.llm_backends.items()
            },
            "nodes": {
                node_id: {
                    "router_host": node.router_host,
                    "router_port": node.router_port,
                    "use_tls": node.use_tls,
                    "auth_token": node.auth_token,
                    "llm_model": node.llm_model,
                    "llm_backend": node.llm_backend,
                    "fold_backend": node.fold_backend,
                    "system_prompt": node.system_prompt,
                    "tools": node.tools,
                }
                for node_id, node in self.nodes.items()
            },
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def get_llm_config_for_node(self, node_id: str) -> LLMBackendConfig | None:
        """
        Get the LLM backend config for a specific node.

        Returns None if node not found or no backend configured.
        """
        node = self.nodes.get(node_id)
        if not node:
            return None

        backend_name = node.llm_backend
        return self.llm_backends.get(backend_name)


def backend_config_to_llm_config(backend: LLMBackendConfig):
    """
    Convert LLMBackendConfig to LLMConfig for the LLMClient.

    This bridges the YAML config format to the LLMClient's expected config.
    """
    from .llm import LLMConfig

    return LLMConfig(
        backend=backend.backend_type,  # type: ignore
        model=backend.default_model,
        max_tokens=backend.max_tokens,
        temperature=backend.temperature,
        api_key=backend.api_key,
        base_url=backend.base_url,
        cc_allowed_tools=backend.cc_allowed_tools,
        cc_fallback_homes=backend.cc_fallback_homes,
        zai_api_key=backend.api_key if backend.backend_type == "zai" else "",
        # Anthropic settings
        anthropic_api_key=backend.api_key if backend.backend_type == "anthropic" else "",
        anthropic_base_url=backend.base_url if backend.backend_type == "anthropic" else "https://api.anthropic.com/v1",
        anthropic_thinking_budget=backend.anthropic_thinking_budget,
        # Thinking/Reasoning settings
        reasoning_effort=backend.reasoning_effort,  # type: ignore
        thinking_level=backend.thinking_level,  # type: ignore
        thinking_budget=backend.thinking_budget,
        include_thoughts=backend.include_thoughts,
        auto_detect_reasoning=backend.auto_detect_reasoning,
        cookie_source=backend.cookie_source,
        # CC subprocess env overrides, thinking mode, binary path, and effort
        cc_env=backend.cc_env,
        cc_thinking=backend.thinking,
        cc_binary=backend.cc_binary,
        cc_effort=backend.cc_effort,
        cc_use_mcp=backend.cc_use_mcp,
        cc_worker_briefing=backend.cc_worker_briefing,
        # Codex settings
        codex_binary=backend.codex_binary,
        codex_subprocess_idle_timeout=backend.codex_subprocess_idle_timeout,
        codex_extra_args=list(backend.codex_extra_args),
        # Mesh harness settings
        harness_python=backend.harness_python,
        harness_backend=backend.harness_backend,
        harness_base_url=backend.harness_base_url,
        harness_api_key=backend.harness_api_key,
        harness_toolset=backend.harness_toolset,
        harness_tools=backend.harness_tools,
        harness_system_prompt_file=backend.harness_system_prompt_file,
        harness_soft_limit=backend.harness_soft_limit,
        harness_controller_mode=backend.harness_controller_mode,
        harness_compaction_threshold_fraction=backend.harness_compaction_threshold_fraction,
        harness_max_phases=backend.harness_max_phases,
        harness_assessor_backend=backend.harness_assessor_backend,
        harness_assessor_model=backend.harness_assessor_model,
        harness_assessor_base_url=backend.harness_assessor_base_url,
        harness_assessor_api_key=backend.harness_assessor_api_key,
        harness_assessor_effort=backend.harness_assessor_effort,
        harness_codex_assessor=backend.harness_codex_assessor,
        harness_codex_assessor_binary=backend.harness_codex_assessor_binary,
        harness_codex_assessor_model=backend.harness_codex_assessor_model,
        harness_codex_assessor_effort=backend.harness_codex_assessor_effort,
    )


def find_config() -> Path | None:
    """
    Find config file in standard locations.

    Search order:
    1. ./mesh.yaml
    2. ~/.hello-world/mesh.yaml
    3. /etc/hello-world/mesh.yaml
    """
    from .paths import real_home
    candidates = [
        Path("mesh.yaml"),
        real_home() / ".hello-world" / "mesh.yaml",
        Path("/etc/hello-world/mesh.yaml"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_config(path: str | Path | None = None) -> MeshConfig:
    """
    Load configuration, searching default locations if path not specified.
    """
    if path is None:
        path = find_config()

    if path is None:
        return MeshConfig()

    return MeshConfig.load(path)
