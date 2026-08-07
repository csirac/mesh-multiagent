"""
Configuration loading for the mesh.

Loads settings from YAML config file with sensible defaults.
Designed with future authentication in mind (placeholder fields).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .isolation import IsolationConfig, IsolationConfigError, IsolationPolicy


# Open/closed model access classification for named LLM backends.
#
# BACKEND_ACCESS_OPEN     — open-weight model (self-hosted, or an open-weight
#                           model served over someone else's API).
# BACKEND_ACCESS_CLOSED   — closed-weight/vendor model.
# ""                      — UNCLASSIFIED.  Every consumer treats this as closed:
#                           the access gate is fail-closed, so forgetting to
#                           classify a new backend can only ever be restrictive.
BACKEND_ACCESS_OPEN = "open"
BACKEND_ACCESS_CLOSED = "closed"
BACKEND_ACCESS_VALUES = frozenset({"", BACKEND_ACCESS_OPEN, BACKEND_ACCESS_CLOSED})


def classify_backend_access(access: Any) -> str:
    """Normalize a raw ``access`` value to ``open`` or ``closed``.

    Fail-closed: anything that is not exactly ``open`` — missing, empty,
    misspelled, or a non-string — classifies as ``closed``.
    """
    value = str(access or "").strip().lower()
    return (
        BACKEND_ACCESS_OPEN
        if value == BACKEND_ACCESS_OPEN
        else BACKEND_ACCESS_CLOSED
    )


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


def load_raw_prompt_file(filename: str, prompts_dir: Path | None = None) -> str:
    """Load a prompt file's raw text *without* appending shared includes.

    ``load_prompt_file`` automatically appends ``channel_policy.md``,
    ``memory.md``, and ``mesh_tools.md`` to any non-shared prompt.  Those
    includes are already part of an agent's standing system prompt (loaded the
    same way from ``system_prompt_file``), so using the include-appending loader
    for text that is injected *in addition to* the system prompt — most notably
    the autonomous-controller operating mandate (plan §10.1) — would duplicate
    ~3K tokens of shared context on every autonomous turn.  Use this loader for
    such per-turn-injected text.

    Args:
        filename: Filename (e.g., "autonomous_controller.txt") or relative path.
        prompts_dir: Override prompts directory (defaults to mesh/prompts/).

    Returns:
        The raw prompt content as a string, or empty string if file not found.
    """
    if prompts_dir is None:
        prompts_dir = PROMPTS_DIR

    prompt_path = prompts_dir / filename
    if not prompt_path.exists():
        return ""

    return prompt_path.read_text().strip()


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
    base_url: str | None = None    # Optional OpenAI-compatible endpoint override
    api_key: str | None = None     # None = use env; "" = no Authorization header
    max_tokens: int = 1024         # Local Qwen may emit hidden-style analysis before final SCORE


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

    # Claude Code account-usage polling is an optional operator dashboard
    # feature.  It must never discover local credentials or make network
    # requests merely because a router was started.
    cc_usage_monitor_enabled: bool = False

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


@dataclass(frozen=True)
class FixedToolParameter:
    """One typed CLI parameter exposed by a router fixed tool."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    cli_flag: str = ""


@dataclass(frozen=True)
class FixedToolConfig:
    """Configuration for an external pipeline launched in a worker slot."""

    name: str
    command: str
    description: str = ""
    timeout_hours: float = 24.0
    parameters: list[FixedToolParameter] = field(default_factory=list)
    phase_markers: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)


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
    # Worker completion/report synthesis timeout, in seconds.  This is kept
    # backend-specific because local vLLM routers can spend much longer in
    # cold long-context prefill than API-hosted routers.
    synthesis_timeout: int = 180

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
    chat_template_kwargs: dict[str, Any] | None = None  # vLLM/OpenAI-compatible chat template controls

    # Reserved for optional cookie-based auth providers. The public release
    # intentionally ships no site-specific cookie adapter.
    cookie_source: str = ""

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

    # Codex subprocess environment overrides.  This is intentionally separate
    # from cc_env: Codex may need its own endpoint and API-key settings when
    # routed through an OpenAI-compatible local provider.
    codex_env: dict[str, str] | None = None

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

    # Open/closed model classification, the model-level twin of the isolation
    # network gate.  "open" means an open-weight model (self-hosted, or an open
    # -weight model served by an API); "closed" means a closed-weight/vendor
    # model.  Classify by the MODEL's provider, not the transport client: a
    # Claude Code or Codex binary pointed at a local open-weight server is
    # "open".  An empty value is UNCLASSIFIED and is treated as closed by
    # every consumer — the gate fails closed by construction.
    access: str = ""

    def __post_init__(self):
        if self.access not in BACKEND_ACCESS_VALUES:
            raise ValueError(
                "llm_backends access must be one of: "
                f"{', '.join(sorted(v for v in BACKEND_ACCESS_VALUES if v))} "
                f"(got {self.access!r})"
            )
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

    def scoped_session_dir(self, state_paths=None) -> str:
        """Session directory for this agent.

        Returns the scoped ``cc_sessions_dir`` when an isolation policy
        supplied one, otherwise the configured/global path unchanged.
        """
        if state_paths is not None:
            return str(state_paths.cc_sessions_dir)
        return self.session_dir


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


@dataclass(frozen=True)
class PevTaskConfig:
    """Resolved phase backends for a phase-selective PEV worker.

    This is intentionally task-type metadata, rather than a backend definition:
    the dispatch layer selects it once and the PEV harness receives only these
    already-resolved backend names.  Phase presence determines the workflow:
    Plan + Execute is full PEV, Plan alone is plan-only, and Execute (with
    optional Verify) is execute mode.

    Optional fields ``worker_system_prompt_file`` and ``worker_instructions_file``
    let a task type carry type-specific prompt content that the agent node
    loads and appends to the assembled system prompt / worker instructions.

    Optional field ``compose_backend`` lets a task type carry a backend name
    for the synchronous ``style_filter`` mesh tool.
    """

    plan: str | None
    execute: str | None
    verify: str | None = None
    worker_system_prompt_file: str | None = None
    worker_instructions_file: str | None = None
    compose_backend: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "plan": self.plan,
            "execute": self.execute,
            "verify": self.verify,
            "worker_system_prompt_file": self.worker_system_prompt_file,
            "worker_instructions_file": self.worker_instructions_file,
            "compose_backend": self.compose_backend,
        }

    @property
    def mode(self) -> str:
        """Infer and validate the harness mode encoded by phase presence."""
        if self.verify and not self.execute:
            raise ValueError(
                "worker task type pev cannot configure verify without execute"
            )
        if self.plan and self.execute:
            return "full"
        if self.plan:
            return "plan"
        if self.execute:
            return "execute"
        raise ValueError(
            "worker task type pev requires at least one of plan or execute"
        )

    @classmethod
    def from_dict(cls, raw: dict) -> "PevTaskConfig":
        """Construct and validate a dict (e.g. custom-launch metadata)."""
        normalized = normalize_pev_task_config(raw)
        if normalized is None:  # Defensive: a dict never normalizes to no policy.
            raise ValueError("worker task type pev must be a mapping")
        return normalized



_TASK_PROMPT_PHASES = frozenset({"plan", "execute", "verify"})


def _normalize_phase_tool_config(
    raw_tools: Any,
    *,
    field_name: str,
    allow_empty_lists: bool,
    reject_commas: bool = False,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Validate and freeze one per-phase prompt tool mapping."""
    if raw_tools is None:
        raw_tools = {}
    if not isinstance(raw_tools, dict):
        raise ValueError(f"prompts.{field_name} must be a mapping")

    phases: list[tuple[str, tuple[str, ...]]] = []
    for raw_phase, raw_names in raw_tools.items():
        phase = str(raw_phase or "").strip().lower()
        if phase not in _TASK_PROMPT_PHASES:
            raise ValueError(
                f"prompts.{field_name} phases must be plan, execute, or verify"
            )
        if not isinstance(raw_names, (list, tuple)):
            raise ValueError(f"prompts.{field_name}.{phase} must be a list")

        names: list[str] = []
        for raw_name in raw_names:
            if not isinstance(raw_name, str):
                raise ValueError(
                    f"prompts.{field_name}.{phase} entries must be strings"
                )
            name = raw_name.strip()
            if not name:
                raise ValueError(
                    f"prompts.{field_name}.{phase} contains an empty tool name"
                )
            if reject_commas and "," in name:
                raise ValueError(
                    f"prompts.{field_name}.{phase} tool names cannot contain commas"
                )
            if name not in names:
                names.append(name)
        if not names and not allow_empty_lists:
            raise ValueError(
                f"prompts.{field_name}.{phase} must name at least one tool"
            )
        phases.append((phase, tuple(names)))
    return tuple(sorted(phases))


def _normalize_prompt_thinking_budget(raw_budget: Any) -> int | None:
    if raw_budget is None:
        return None
    if isinstance(raw_budget, bool):
        raise ValueError("prompts.thinking_budget must be a positive integer")
    try:
        thinking_budget = int(raw_budget)
    except (TypeError, ValueError) as exc:
        raise ValueError("prompts.thinking_budget must be a positive integer") from exc
    if not 1 <= thinking_budget <= 1_000_000:
        raise ValueError("prompts.thinking_budget must be between 1 and 1000000")
    return thinking_budget


@dataclass(frozen=True)
class TaskPromptConfig:
    """Task-level prompt bundle shared by direct, synchronous, and PEV paths.

    Prompt configuration deliberately lives beside ``pev`` rather than inside
    it: ordinary workers and synchronous tools must be able to consume the
    same canonical domain instructions without opting into PEV.
    """

    worker_system_prompt_file: str | None = None
    base_instructions_file: str | None = None
    sync_instructions_file: str | None = None
    plan_instructions_file: str | None = None
    execute_instructions_file: str | None = None
    verify_instructions_file: str | None = None
    sync_backend: str | None = None
    thinking_budget: int | None = None
    phase_mesh_tools: tuple[tuple[str, tuple[str, ...]], ...] = ()
    phase_harness_tools: tuple[tuple[str, tuple[str, ...]], ...] = ()
    verify_read_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "worker_system_prompt_file": self.worker_system_prompt_file,
            "base_instructions_file": self.base_instructions_file,
            "sync_instructions_file": self.sync_instructions_file,
            "plan_instructions_file": self.plan_instructions_file,
            "execute_instructions_file": self.execute_instructions_file,
            "verify_instructions_file": self.verify_instructions_file,
            "sync_backend": self.sync_backend,
            "thinking_budget": self.thinking_budget,
            "phase_mesh_tools": {
                phase: list(tool_names)
                for phase, tool_names in self.phase_mesh_tools
            },
            "phase_harness_tools": {
                phase: list(tool_names)
                for phase, tool_names in self.phase_harness_tools
            },
            "verify_read_only": self.verify_read_only,
        }
    @classmethod
    def from_dict(cls, raw: dict) -> "TaskPromptConfig":
        """Construct from a dict (e.g. custom-launch metadata)."""
        def _opt(s: str | None) -> str | None:
            if s is None:
                return None
            v = str(s).strip()
            return v or None
        budget = _normalize_prompt_thinking_budget(raw.get("thinking_budget"))
        phase_mesh_tools = _normalize_phase_tool_config(
            raw.get("phase_mesh_tools"),
            field_name="phase_mesh_tools",
            allow_empty_lists=True,
        )
        phase_harness_tools = _normalize_phase_tool_config(
            raw.get("phase_harness_tools"),
            field_name="phase_harness_tools",
            allow_empty_lists=False,
            reject_commas=True,
        )
        verify_read_only = raw.get("verify_read_only", False)
        if not isinstance(verify_read_only, bool):
            raise ValueError("prompts.verify_read_only must be a boolean")
        return cls(
            worker_system_prompt_file=_opt(raw.get("worker_system_prompt_file")),
            base_instructions_file=_opt(raw.get("base_instructions_file")),
            sync_instructions_file=_opt(raw.get("sync_instructions_file")),
            plan_instructions_file=_opt(raw.get("plan_instructions_file")),
            execute_instructions_file=_opt(raw.get("execute_instructions_file")),
            verify_instructions_file=_opt(raw.get("verify_instructions_file")),
            sync_backend=_opt(raw.get("sync_backend")),
            thinking_budget=budget,
            phase_mesh_tools=phase_mesh_tools,
            phase_harness_tools=phase_harness_tools,
            verify_read_only=verify_read_only,
        )

    def tools_for_phase(self, phase: str) -> tuple[str, ...]:
        for configured_phase, tool_names in self.phase_mesh_tools:
            if configured_phase == phase:
                return tool_names
        return ()

    def harness_tools_for_phase(self, phase: str) -> tuple[str, ...]:
        for configured_phase, tool_names in self.phase_harness_tools:
            if configured_phase == phase:
                return tool_names
        return ()


WorkerTaskTypeDefinition = dict[str, Any]


def normalize_pev_task_config(raw_pev: Any) -> PevTaskConfig | None:
    """Validate an optional ``worker_task_types.<type>.pev`` block.

    ``verify: null`` (and the legacy string ``"none"``) deliberately means
    that an Execute-capable worker ends after Execute.  Valid phase shapes are
    Plan + Execute (full PEV), Plan only, or Execute with optional Verify.
    Silently degrading an empty or Verify-only policy into an ordinary worker
    would be an unsafe configuration error.
    """
    if raw_pev is None:
        return None
    if isinstance(raw_pev, PevTaskConfig):
        raw_pev = raw_pev.as_dict()
    if not isinstance(raw_pev, dict):
        raise ValueError("worker task type pev must be a mapping")

    plan = str(raw_pev.get("plan") or "").strip() or None
    execute = str(raw_pev.get("execute") or "").strip() or None

    raw_verify = raw_pev.get("verify")
    if raw_verify is None:
        verify = None
    else:
        verify_text = str(raw_verify).strip()
        verify = None if not verify_text or verify_text.lower() == "none" else verify_text

    raw_wsp = raw_pev.get("worker_system_prompt_file")
    worker_system_prompt_file = str(raw_wsp).strip() if raw_wsp else None
    raw_wi = raw_pev.get("worker_instructions_file")
    worker_instructions_file = str(raw_wi).strip() if raw_wi else None

    raw_cb = raw_pev.get("compose_backend")
    compose_backend = str(raw_cb).strip() if raw_cb else None

    normalized = PevTaskConfig(
        plan=plan, execute=execute, verify=verify,
        worker_system_prompt_file=worker_system_prompt_file,
        worker_instructions_file=worker_instructions_file,
        compose_backend=compose_backend,
    )
    # Keep one canonical validity rule for YAML, trusted custom launches, and
    # AgentNode's defensive mode selection.
    normalized.mode
    return normalized


def normalize_task_prompt_config(raw_prompts: Any) -> TaskPromptConfig | None:
    """Validate an optional ``worker_task_types.<type>.prompts`` block."""
    if raw_prompts is None:
        return None
    if isinstance(raw_prompts, TaskPromptConfig):
        raw_prompts = raw_prompts.as_dict()
    if not isinstance(raw_prompts, dict):
        raise ValueError("worker task type prompts must be a mapping")
    return TaskPromptConfig.from_dict(raw_prompts)


def _promote_legacy_prompt_config(
    prompts: TaskPromptConfig | None,
    pev: PevTaskConfig | None,
) -> TaskPromptConfig | None:
    """Map legacy writing prompt fields into the task-level prompt bundle."""
    if pev is None or not any((
        pev.worker_system_prompt_file,
        pev.worker_instructions_file,
        pev.compose_backend,
    )):
        return prompts

    legacy_values = {
        "worker_system_prompt_file": pev.worker_system_prompt_file,
        "execute_instructions_file": pev.worker_instructions_file,
        "sync_backend": pev.compose_backend,
    }
    if prompts is not None:
        for field_name, legacy_value in legacy_values.items():
            configured_value = getattr(prompts, field_name)
            if (
                legacy_value is not None
                and configured_value is not None
                and legacy_value != configured_value
            ):
                raise ValueError(
                    f"legacy pev.{field_name} conflicts with prompts.{field_name}"
                )

    base = prompts or TaskPromptConfig()
    return TaskPromptConfig(
        worker_system_prompt_file=(
            base.worker_system_prompt_file or pev.worker_system_prompt_file
        ),
        base_instructions_file=base.base_instructions_file,
        sync_instructions_file=base.sync_instructions_file,
        plan_instructions_file=base.plan_instructions_file,
        execute_instructions_file=(
            base.execute_instructions_file or pev.worker_instructions_file
        ),
        verify_instructions_file=base.verify_instructions_file,
        sync_backend=base.sync_backend or pev.compose_backend,
        thinking_budget=base.thinking_budget,
        phase_mesh_tools=base.phase_mesh_tools,
        phase_harness_tools=base.phase_harness_tools,
        verify_read_only=base.verify_read_only,
    )


def _merge_task_type_value(base: Any, override: Any) -> Any:
    """Merge one override value over its default, recursing into mappings.

    ``None`` in the override means "not specified — inherit".  This is what
    makes a partial declaration work at field granularity: an agent block that
    says only ``{backend: X}`` for a type leaves ``description``/``pev``/
    ``prompts`` untouched, and an explicit ``backend: null`` falls back to the
    default backend for that type rather than blanking it.
    """
    if override is None:
        return base
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _merge_task_type_value(merged.get(key), value)
        return merged
    return override


def merge_worker_task_types(
    defaults: Any,
    declared: Any,
) -> dict[str, Any]:
    """Deep-merge an agent's declared task types over the fleet defaults.

    YAML anchor merge is shallow: a ``worker_task_types`` key in an agent block
    shadows the anchor's entire map, so an agent that wants to change one
    backend has to restate every type.  This merge restores inheritance at two
    levels — per type (an undeclared type keeps its default entry) and per
    field within a type (an undeclared field keeps its default value).

    Both arguments are RAW pre-normalization mappings, and the result is a new
    mapping: neither input is mutated.  That matters because the YAML alias
    ``worker_task_types: *worker_task_types_defaults`` makes an inheriting
    node's declared map the *same object* as the defaults.
    """
    if not isinstance(defaults, dict) or not defaults:
        return dict(declared) if isinstance(declared, dict) else {}
    if not isinstance(declared, dict):
        return copy.deepcopy(defaults)

    merged: dict[str, Any] = copy.deepcopy(defaults)
    for task_type, definition in declared.items():
        base = merged.get(task_type)
        # A string binding (``name: backend``) is the legacy shorthand and
        # carries no fields to merge — it replaces the entry outright.
        if isinstance(definition, dict) and isinstance(base, dict):
            merged[task_type] = _merge_task_type_value(base, definition)
        elif definition is None:
            continue
        else:
            merged[task_type] = copy.deepcopy(definition)
    return merged


def normalize_worker_task_types(
    raw_task_types: Any,
) -> dict[str, WorkerTaskTypeDefinition]:
    """Normalize task-type bindings to the rich config representation.

    The preferred YAML form is ``name: {backend, description}``.  Plain
    ``name: backend`` entries remain accepted so older configs and callers can
    upgrade without a flag day; they receive an empty description and render
    in router guidance by name only.
    """
    if not isinstance(raw_task_types, dict):
        return {}

    normalized: dict[str, WorkerTaskTypeDefinition] = {}
    for raw_name, raw_definition in raw_task_types.items():
        name = str(raw_name or "").strip()
        if not name:
            continue

        if isinstance(raw_definition, str):
            backend = raw_definition.strip()
            description = ""
            pev = None
            prompts = None
        elif isinstance(raw_definition, dict):
            backend = str(raw_definition.get("backend") or "").strip()
            description = str(raw_definition.get("description") or "").strip()
            try:
                pev = normalize_pev_task_config(raw_definition.get("pev"))
                prompts = normalize_task_prompt_config(
                    raw_definition.get("prompts")
                )
                prompts = _promote_legacy_prompt_config(prompts, pev)
            except ValueError as exc:
                raise ValueError(f"worker_task_types.{name}: {exc}") from exc
        else:
            continue

        if not backend:
            continue
        definition: WorkerTaskTypeDefinition = {
            "backend": backend,
            "description": description,
        }
        if pev is not None:
            definition["pev"] = pev
        if prompts is not None:
            definition["prompts"] = prompts
        normalized[name] = definition
    return normalized


ENTITY_RESOLUTION_MODES = frozenset({"off", "shadow", "write"})


def normalize_entity_resolution_mode(
    mode: str | None,
    *,
    legacy_enabled: bool = False,
) -> str:
    """Normalize the Increment 3 entity authority mode.

    ``entity_resolution_enabled`` remains accepted for old YAML and callers.
    A true legacy flag promotes the default/off mode to ``write``; an explicit
    ``shadow`` or ``write`` mode takes precedence.
    """
    normalized = str(mode or "off").strip().lower()
    if normalized not in ENTITY_RESOLUTION_MODES:
        raise ValueError(
            "entity_resolution_mode must be one of: off, shadow, write"
        )
    if normalized == "off" and legacy_enabled:
        return "write"
    return normalized


def resolve_self_curation_mode(node: "NodeConfig") -> str:
    """Return the effective curation authority for ``node``.

    ``entity_resolution_mode="off"`` always wins over the master flag, and the
    master flag off means no curation turn regardless of mode.
    """
    resolution = normalize_entity_resolution_mode(
        getattr(node, "entity_resolution_mode", "off"),
        legacy_enabled=bool(getattr(node, "entity_resolution_enabled", False)),
    )
    if resolution == "off":
        return "off"
    if not getattr(node, "entity_self_curation_enabled", False):
        return "off"
    requested = str(
        getattr(node, "entity_self_curation_mode", "") or ""
    ).strip().lower()
    if not requested:
        return resolution
    if requested not in ENTITY_RESOLUTION_MODES:
        raise ValueError(
            "entity_self_curation_mode must be one of: off, shadow, write"
        )
    if requested == "write" and resolution != "write":
        # Never escalate beyond the registry's own authority.
        return resolution
    return requested


def validate_self_curation_enrollment(node: "NodeConfig") -> list[str]:
    """Return one error string per unmet self-curation enrollment condition.

    Enrollment requires Memory Formation v3 plus a rolling-window full router,
    an existing standing digest containing all seven constitutional sections, a
    tiktoken-backed ``estimate_tokens()``, and ``project_maps_enabled=false``
    so the legacy map writer does not become a second post-formation curator.
    """
    if resolve_self_curation_mode(node) == "off":
        return []

    errors: list[str] = []
    if not getattr(node, "memory_formation_v3_enabled", False):
        errors.append(
            "entity self-curation requires memory_formation_v3_enabled=true"
        )
    if str(getattr(node, "context_mode", "rolling-window")) != "rolling-window":
        errors.append(
            "entity self-curation requires context_mode=rolling-window "
            "(cc-session/RouterCC has no _call_router_full path)"
        )
    if str(getattr(node, "router_mode", "classifier")) != "full":
        errors.append(
            "entity self-curation requires router_mode=full"
        )
    if getattr(node, "project_maps_enabled", True):
        errors.append(
            "entity self-curation requires project_maps_enabled=false "
            "(the legacy map writer would be a second post-formation curator)"
        )
    if not getattr(node, "standing_digest_enabled", False):
        errors.append(
            "entity self-curation requires standing_digest_enabled=true"
        )
    if int(getattr(node, "standing_digest_budget_tokens", 0)) < 1:
        errors.append("standing_digest_budget_tokens must be at least 1")

    digest_path = str(getattr(node, "standing_digest_path", "") or "")
    if not digest_path:
        errors.append(
            "entity self-curation requires standing_digest_path"
        )
    else:
        expanded = os.path.expanduser(digest_path)
        if not os.path.exists(expanded):
            errors.append(
                f"standing digest file not found at {expanded}"
            )
        else:
            try:
                from .memory.curation import digest_section_errors

                with open(expanded) as handle:
                    section_errors = digest_section_errors(handle.read())
            except OSError as exc:
                section_errors = [f"digest unreadable: {exc}"]
            errors.extend(
                f"standing digest: {item}" for item in section_errors
            )

    if resolve_self_curation_mode(node) == "write":
        try:
            from .llm import _encoder
        except Exception:  # pragma: no cover - import-time backend problems
            _encoder = None
        if _encoder is None:
            errors.append(
                "write-mode self-curation requires a tiktoken encoder; the "
                "word-heuristic estimate_tokens() fallback is not a true "
                "token limit"
            )
    return errors


@dataclass
class NodeConfig:
    """Configuration for a node (user or agent)."""
    id: str                          # e.g., "user:operator" or "agent:researcher"
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
    # DEPRECATED flat form — superseded by the nested ``isolation`` block below.
    # Retained for one deprecation cycle; setting both forms is a config error.
    sandboxed: bool = False                       # Enable bwrap sandboxing
    allowed_dirs: list[str] = field(default_factory=list)  # Writable directories
    allow_network: bool = True                    # Allow network access in sandbox

    # Per-agent filesystem/capability isolation (see mesh/isolation.py).
    # Absent or ``enabled: false`` selects the legacy path: ~/.mesh state,
    # unfiltered tools, and no policy checks.
    isolation: "IsolationConfig | None" = None

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
    use_relevance_router_for_channels: bool = False  # RouterV2 channel gate: LLM relevance instead of hard @mention

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

    # Optional direct-router client used only for an explicit leading @deep
    # override. Phase 1 has no model-driven escalation or harness support.
    router_deep_backend: str | None = None
    router_deep_enabled: bool = False

    # History management — unified fields for both router and worker.
    # When summarization is disabled, history is a simple rolling window:
    # oldest turns are dropped when the window exceeds hard_limit_tokens.
    history_summarization_enabled: bool = False       # off = rolling window only
    history_soft_limit_tokens: int = 70_000           # rolling window cap
    history_hard_limit_tokens: int = 90_000           # hard cap (drop oldest turns)
    history_window_tokens: int | None = None           # rolling window budget W (default: soft_limit // 2)
    watchdog_interval_minutes: int = 0                # worker watchdog check-in interval (0 = disabled)
    worker_context_window_tokens: int = 25_000         # token budget for worker context snapshot
    worker_in_flight_token_limit: int = 150_000       # safety valve for tool loops (independent)
    max_concurrent_workers: int = 1                   # worker slots per agent; 1 preserves legacy behavior
    # Fail closed on missing/degenerate router briefs.  Provenance validation
    # rejects trigger-content fallback independently of this length backstop.
    min_worker_brief_chars: int = 120
    # Deprecated compatibility fields. Router LLMs cannot create sticky
    # backend overrides, and stale allowlists are accepted only so older YAML
    # still loads. Ordinary per-launch routing uses ``worker_task_types``.
    worker_backend_override_enabled: bool = False
    worker_backends_allowed: list[str] = field(default_factory=list)
    # Task-shape indirection for worker launches. Routers select a type; this
    # configuration maps it to the backend. The configured ``llm_backend`` is
    # still the fallback/default, while explicit user backend requests are the
    # only route that bypasses this mapping.
    worker_task_types: dict[str, WorkerTaskTypeDefinition] = field(
        default_factory=dict
    )
    # Open/closed model access gate — the model-level twin of the isolation
    # network gate.  True (the default) is today's behaviour: this agent may
    # dispatch a worker to any configured backend.  False makes the agent
    # structurally incapable of running a closed-weight model: every dispatch
    # branch (task type, verbatim user override, custom staged selection) and
    # every phase backend (pev.plan/execute/verify/compose_backend,
    # prompts.sync_backend) is checked, and a closed backend is REFUSED rather
    # than silently replaced by the configured default.
    worker_closed_models: bool = True
    # Inject the agent's published standing digest into every worker briefing.
    # The digest is background/index context; workers retrieve full records and
    # essays through memory tools.  Token-sensitive agents may disable this.
    worker_digest_injection: bool = True
    # Persist full completed-worker traces privately for skill_draft evidence.
    # This archive is separate from router history and prompt-capture logs.
    worker_trace_persist: bool = True

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
    memory_v3_window_size: int = 60                  # extractor window
    memory_v3_overlap: int = 20                      # window overlap
    memory_v3_defer_tail: int = 10                   # in-window trailing turns deferred
    memory_v3_model: str = "deepseek-v4-flash"       # extraction model
    memory_v3_parse_failure_fallback_threshold: int = 3  # placeholder after N consecutive failures

    # Deprecated compatibility flag. Option A-prime made recall-oriented
    # extraction the sole formation contract; live and fold paths ignore false.
    memory_formation_lowbar: bool = True

    # Durable entity registry / correction API. Schema migration is always
    # additive. ``shadow`` enables formation parsing/telemetry without writes;
    # only ``write`` exposes mutation tools and authorizes persistence.
    entity_resolution_mode: str = "off"
    entity_registry_injection_cap: int = 1000
    # Output ceiling for one entity-enabled formation request. This bounds
    # reasoning tokens AND content on a reasoning backend, so it must cover
    # both. The original 1,200 was content-only planning arithmetic: measured
    # against a real 60-turn Bob window on deepseek-v4-flash, one call spends
    # ~31.6K reasoning + ~8.4K content (18 records). At 1,200 the whole budget
    # went to reasoning, the API returned an empty completion, and every
    # formation run failed contract parsing. Formation floors this at the core
    # memory_v3 max_tokens; see LLMExtractorV1.request_max_tokens.
    entity_formation_max_tokens: int = 48000
    # Deprecated compatibility input/output. __post_init__ derives this from
    # entity_resolution_mode after accepting old boolean-only configurations.
    entity_resolution_enabled: bool = False
    entity_activation_window_threshold: int = 3

    # Entity/group/digest self-curation (docs/plans/entity-self-curation.md).
    # The master trigger gate: when True, every successful formation batch
    # enqueues one internal curation turn on the agent's own router backend.
    # An agent in entity_resolution_mode="write" with this flag off keeps
    # interactive entity_link_correct and loses only the automatic turn.
    entity_self_curation_enabled: bool = False
    # Authority for the curation turn.  Empty string inherits
    # entity_resolution_mode; "off" always wins over the master flag.
    entity_self_curation_mode: str = ""
    # Phase 2: group tools + deterministic bridge-evidence activation.
    entity_self_curation_groups_enabled: bool = False
    # Hard digest ceiling, measured with mesh.llm.estimate_tokens().
    standing_digest_budget_tokens: int = 32000
    # Report-only stale pending-group threshold (consecutive curation batches).
    curation_stale_group_batches: int = 50
    # Consecutive failed curation turns before ERROR/agent_status alert.
    curation_failure_alert_threshold: int = 5
    # Phase 3 agent-driven backfill.  Fires once after the startup formation
    # chain when there is uncurated history; also reachable on demand through
    # the entity_backfill tool.  Both knobs are bounded on purpose: one
    # invocation must never be able to curate the whole history.
    entity_self_curation_backfill_on_startup: bool = True
    entity_self_curation_backfill_max_batches: int = 50
    # Memories per backfill slice ("fixed-size slices", §9 Phase 3).
    entity_self_curation_backfill_slice_size: int = 10
    # Essay generation folded into curation: after a curation turn commits,
    # write the essay for any entity that is active and has none.  Default OFF
    # so enabling is a deliberate per-agent act; the post-hoc batch script
    # stays the behaviour when this is off.  Honours entity_self_curation_mode
    # — "shadow" validates the essay and rolls back instead of writing.
    entity_self_curation_essays_enabled: bool = False
    # Essays written per curation turn.  Bounded on purpose: a backlog drains
    # over successive turns instead of stalling one turn behind N LLM calls.
    entity_self_curation_essays_max_per_turn: int = 1

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

    # Project-map injection: when False, render_relevant_maps_block() and
    # render_maps_block() return "" — no map content enters any prompt pathway.
    # Maps remain in the DB; only injection is suppressed.  (Decision 29:
    # project maps deprecated in favour of the essay layer.)
    project_maps_enabled: bool = True

    memory_search_mode: str = "hybrid"  # "embedding" | "lexical" | "hybrid"

    # Essay layer (Phase 2, per-agent-standing-digest spec decisions 26-32):
    # essay_fold_enabled gates essay-edit tooling in the fold driver.
    # essay_recurrence_threshold: entity must appear in this many distinct
    # fold windows before a new essay is created (tuning knob §10b.1).
    # essay_token_budget: max tokens per individual essay (§10b.4).
    essay_fold_enabled: bool = False
    essay_recurrence_threshold: int = 3
    essay_token_budget: int = 4000

    # Essay retrieval (Phase 4): when True, essay_get and essay_list are
    # dynamically added to the agent's enabled_tools at startup, giving
    # live agents pull-based access to curated essays.  essay_edit is
    # NEVER exposed — it remains fold-engine-only.
    essays_retrieval_enabled: bool = False

    # Essay auto-maintain: when True, the nightly fold autonomously runs
    # essay CREATE/PATCH/hygiene after digest finalization.  Defaults OFF
    # so the first pass is human-gated via run_meta_review.py --apply.
    essay_auto_maintain: bool = False

    # Fold driver LLM backend (standing-digest offline fold).
    # Resolved by _resolve_fold_backend() in the fold driver.
    fold_backend: str = "mesh-harness-qwen"

    # Autonomous agent mode.  When True the agent may act as an autonomous
    # project controller: the dossier tools are added to enabled_tools and the
    # autonomous_controller prompt becomes available for per-turn injection.
    # The mandate is injected only on turns whose trigger carries trusted
    # autonomous-session metadata (plan §10.1) — enrollment alone does not put
    # ordinary conversation turns under the autonomous operating mandate.
    autonomous_agent_mode_enabled: bool = False
    autonomous_controller_prompt_file: str = "autonomous_controller.txt"
    # Optional execute-only mandate for worker-report continuations.  Empty
    # retains the full wake mandate for backwards-compatible deployments.
    autonomous_controller_continuation_prompt_file: str = ""
    # Initial autonomous PLAN turns may use the deep router; report-driven
    # execute/close continuations always remain on the light router.
    autonomous_plan_backend: str = "light"
    # Project entity keys this agent controls, e.g. ["project:mesh-infra"].
    autonomous_projects: list[str] = field(default_factory=list)
    autonomous_max_workers_per_session: int = 2
    # Active-mode pacing.  A project armed with `/auto active <project> on`
    # schedules its next session automatically at closeout; this is the minimum
    # gap between one session ending and the next beginning, and also the
    # minimum separation enforced between wakes across an agent's projects so
    # two of its sessions never overlap.  Paced, not continuous.
    autonomous_active_gap_minutes: int = 60
    # The recursive autonomous controller (mesh/autonomous_controller.py) is a
    # separate pilot harness, not the control path.  Plan §10.3 forbids a
    # second planner competing with the agent's own ReAct loop, so the
    # autonomous_controller_run tool is unreachable unless an operator opts a
    # single agent into it explicitly.
    autonomous_recursive_controller_enabled: bool = False

    memory_get_payload_max_chars: int = 6000
    memory_search_default_k: int = 5

    # Router V2 full mode settings
    # "full" = conversational agent with tools; "classifier" = legacy thin classifier
    router_mode: str = "classifier"
    router_max_iters: int = 50  # Max tool-loop iterations for full router
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
        self.worker_task_types = normalize_worker_task_types(
            self.worker_task_types
        )
        # A truthy string ("false") would silently disable the gate, so demand
        # a real bool rather than coercing.
        if not isinstance(self.worker_closed_models, bool):
            raise ValueError(
                "worker_closed_models must be a boolean "
                f"(got {self.worker_closed_models!r})"
            )
        self.entity_resolution_mode = normalize_entity_resolution_mode(
            self.entity_resolution_mode,
            legacy_enabled=self.entity_resolution_enabled,
        )
        self.entity_resolution_enabled = (
            self.entity_resolution_mode == "write"
        )
        if self.entity_registry_injection_cap < 1:
            raise ValueError("entity_registry_injection_cap must be at least 1")
        if self.entity_formation_max_tokens < 1:
            raise ValueError("entity_formation_max_tokens must be at least 1")
        requested_curation = str(self.entity_self_curation_mode or "").strip().lower()
        if requested_curation and requested_curation not in ENTITY_RESOLUTION_MODES:
            raise ValueError(
                "entity_self_curation_mode must be one of: off, shadow, write"
            )
        self.entity_self_curation_mode = requested_curation
        if self.standing_digest_budget_tokens < 1:
            raise ValueError("standing_digest_budget_tokens must be at least 1")
        if self.curation_stale_group_batches < 1:
            raise ValueError("curation_stale_group_batches must be at least 1")
        if self.curation_failure_alert_threshold < 1:
            raise ValueError(
                "curation_failure_alert_threshold must be at least 1"
            )
        if self.entity_self_curation_backfill_max_batches < 1:
            raise ValueError(
                "entity_self_curation_backfill_max_batches must be at least 1"
            )
        if self.entity_self_curation_backfill_slice_size < 1:
            raise ValueError(
                "entity_self_curation_backfill_slice_size must be at least 1"
            )
        if self.entity_self_curation_essays_max_per_turn < 1:
            raise ValueError(
                "entity_self_curation_essays_max_per_turn must be at least 1"
            )
        requested_plan_backend = str(
            self.autonomous_plan_backend or ""
        ).strip().lower()
        self.autonomous_plan_backend = (
            requested_plan_backend
            if requested_plan_backend in {"light", "deep"}
            else "light"
        )
        # Expand environment variable references in auth_token
        if self.auth_token and self.auth_token.startswith("${") and self.auth_token.endswith("}"):
            env_var = self.auth_token[2:-1]
            self.auth_token = os.environ.get(env_var, "")
        # Accept a plain mapping for ``isolation`` so directly-constructed
        # NodeConfig objects (tests, programmatic callers) behave like YAML.
        if self.isolation is not None and not isinstance(self.isolation, IsolationConfig):
            self.isolation = IsolationConfig.from_dict(self.isolation)

    def resolve_isolation_policy(self) -> IsolationPolicy:
        """Normalize this node's isolation settings into one policy.

        Precedence and conflict rules:

        * A nested ``isolation`` block is authoritative.
        * Setting both the nested block and the deprecated flat
          ``sandboxed``/``allowed_dirs`` fields is rejected rather than
          resolved, because either guess would silently be someone's
          security boundary.
        * With only the flat fields set, they are normalized into a
          compatibility policy so the old configuration keeps working.
        * With neither, the legacy (disabled) policy is returned and no
          filesystem work is done.
        """
        legacy_flat_set = bool(self.sandboxed) or bool(self.allowed_dirs)

        if self.isolation is not None and self.isolation.enabled:
            if legacy_flat_set:
                raise IsolationConfigError(
                    f"{self.id}: both the nested 'isolation' block and the "
                    "deprecated flat 'sandboxed'/'allowed_dirs' fields are set — "
                    "remove the flat fields; they are superseded by 'isolation'"
                )
            return IsolationPolicy.from_config(
                self.isolation, source=f"config:{self.id}"
            )

        if self.isolation is not None and legacy_flat_set:
            raise IsolationConfigError(
                f"{self.id}: both the nested 'isolation' block and the "
                "deprecated flat 'sandboxed'/'allowed_dirs' fields are set — "
                "remove the flat fields; they are superseded by 'isolation'"
            )

        if legacy_flat_set:
            return self._legacy_flat_policy()

        return IsolationPolicy.legacy(source=f"config:{self.id}")

    def _legacy_flat_policy(self) -> IsolationPolicy:
        """Normalize deprecated flat sandbox fields into a policy.

        The flat form never relocated state, so ``state_root`` stays at the
        historical global root and the declared roots only constrain tools.
        """
        from .isolation import FilesystemMode, resolve_workspace_path
        from .paths import MESH_DIR

        roots: list[Path] = []
        for raw in self.allowed_dirs or []:
            try:
                resolved = resolve_workspace_path(raw)
            except (OSError, RuntimeError) as exc:
                raise IsolationConfigError(
                    f"{self.id}: allowed_dirs entry {raw!r} cannot be resolved: {exc}"
                ) from exc
            if resolved not in roots:
                roots.append(resolved)

        return IsolationPolicy(
            enabled=bool(self.sandboxed),
            workspace=roots[0] if roots else None,
            workspaces=tuple(roots),
            state_root=Path(MESH_DIR),
            filesystem_mode=FilesystemMode.WORKSPACE_WRITE,
            protect_state=False,
            allow_network=bool(self.allow_network),
            allowed_network_tools=None,
            allowed_credential_tools=None,
            source=f"legacy-flat:{self.id}",
        )


def _load_merged_config_data(path: str | Path) -> dict[str, Any]:
    """Load mesh YAML and apply the split-backend overlay."""
    path = Path(path)
    if not path.exists():
        return {}

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    backends_path = path.parent / "backends.yaml"
    if backends_path.exists():
        with open(backends_path) as f:
            backends_data = yaml.safe_load(f) or {}
        if not isinstance(backends_data, dict):
            raise ValueError(
                f"{backends_path} must contain a YAML mapping of backend names"
            )
        mesh_backends = data.get("llm_backends", {}) or {}
        if not isinstance(mesh_backends, dict):
            raise ValueError(
                f"{path} llm_backends must contain a YAML mapping of backend names"
            )
        merged = dict(backends_data)
        # Local entries in mesh.yaml intentionally override shared split-file
        # definitions. This preserves the old single-file precedence model.
        merged.update(mesh_backends)
        data["llm_backends"] = merged
    return data


def load_llm_backends(path: str | Path) -> dict[str, LLMBackendConfig]:
    """Load named backends without constructing node configuration.

    Harness launchers use this path because they need only one backend.  Keeping
    node parsing out of backend lookup also lets a long-lived PEV parent launch
    Verify after Execute changes ``NodeConfig`` or adds new node YAML fields.
    """
    raw_backends = _load_merged_config_data(path).get("llm_backends", {}) or {}
    if not isinstance(raw_backends, dict):
        raise ValueError(
            f"{Path(path)} llm_backends must contain a YAML mapping of backend names"
        )
    return {
        backend_id: LLMBackendConfig(**(backend_data or {}))
        for backend_id, backend_data in raw_backends.items()
    }


@dataclass
class MeshConfig:
    """Top-level configuration."""
    router: RouterConfig = field(default_factory=RouterConfig)
    nodes: dict[str, NodeConfig] = field(default_factory=dict)
    llm_backends: dict[str, LLMBackendConfig] = field(default_factory=dict)
    fixed_tools: dict[str, FixedToolConfig] = field(default_factory=dict)
    # The fleet-wide ``worker_task_types_defaults`` anchor, normalized.  Each
    # node's effective map is this map deep-merged under whatever the node
    # declares; kept here so the merge is inspectable after load.
    worker_task_types_defaults: dict[str, WorkerTaskTypeDefinition] = field(
        default_factory=dict
    )

    @classmethod
    def load(cls, path: str | Path) -> MeshConfig:
        """Load configuration from a YAML file."""
        return cls.from_dict(_load_merged_config_data(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeshConfig:
        """Create config from a dictionary."""
        router_data = data.get("router", {})
        router = RouterConfig(**router_data)

        # Parse LLM backends
        llm_backends = {}
        for backend_id, backend_data in data.get("llm_backends", {}).items():
            llm_backends[backend_id] = LLMBackendConfig(**backend_data)

        fixed_tools: dict[str, FixedToolConfig] = {}
        for tool_name, tool_data in data.get("fixed_tools", {}).items():
            tool_data = dict(tool_data or {})
            parameter_data = tool_data.pop("parameters", {}) or {}
            parameters = [
                FixedToolParameter(name=parameter_name, **(details or {}))
                for parameter_name, details in parameter_data.items()
            ]
            fixed_tools[tool_name] = FixedToolConfig(
                name=tool_name,
                parameters=parameters,
                **tool_data,
            )

        # Fleet-wide worker task-type defaults.  A node that declares no
        # worker_task_types keeps today's behaviour exactly (empty map, or
        # whatever a YAML anchor copied in); a node that declares a PARTIAL
        # block inherits every type and field it did not name.
        raw_task_type_defaults = data.get("worker_task_types_defaults") or {}
        if not isinstance(raw_task_type_defaults, dict):
            raise ValueError(
                "worker_task_types_defaults must be a YAML mapping of task types"
            )

        nodes = {}
        for node_id, node_data in data.get("nodes", {}).items():
            # Handle nodes defined with no properties (value is None)
            if node_data is None:
                node_data = {}
            node_data["id"] = node_id

            # Merge only when the node actually declares the key.  Merging
            # unconditionally would hand the fleet map to lightweight nodes
            # that deliberately have none today.
            if "worker_task_types" in node_data:
                node_data["worker_task_types"] = merge_worker_task_types(
                    raw_task_type_defaults,
                    node_data.get("worker_task_types"),
                )
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

            # Parse the nested isolation block. Reject the nested/flat conflict
            # here as well as in resolve_isolation_policy() so a bad file fails
            # at load time rather than at agent start.
            if "isolation" in node_data and node_data["isolation"] is not None:
                legacy_isolation_keys = [
                    key for key in ("sandboxed", "allowed_dirs")
                    if key in node_data
                ]
                if not isinstance(node_data["isolation"], IsolationConfig):
                    try:
                        node_data["isolation"] = IsolationConfig.from_dict(
                            node_data["isolation"]
                        )
                    except IsolationConfigError as exc:
                        raise IsolationConfigError(f"{node_id}: {exc}") from exc
                if legacy_isolation_keys:
                    raise IsolationConfigError(
                        f"{node_id}: both the nested 'isolation' block and the "
                        "deprecated flat 'sandboxed'/'allowed_dirs' fields are "
                        "set — remove the flat fields; they are superseded by "
                        "'isolation'"
                    )

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

        for node_id, node in nodes.items():
            if not node.router_deep_enabled:
                continue
            if not node.router_deep_backend:
                raise ValueError(
                    f"{node_id}: router_deep_backend is required when "
                    "router_deep_enabled is true"
                )
            if node.router_deep_backend not in llm_backends:
                raise ValueError(
                    f"{node_id}: router_deep_backend "
                    f"{node.router_deep_backend!r} is not present in "
                    "llm_backends"
                )

        return cls(
            router=router,
            nodes=nodes,
            llm_backends=llm_backends,
            fixed_tools=fixed_tools,
            worker_task_types_defaults=normalize_worker_task_types(
                raw_task_type_defaults
            ),
        )

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
                    "access": backend.access,
                    "api_key": backend.api_key,
                    "base_url": backend.base_url,
                    "default_model": backend.default_model,
                    "max_tokens": backend.max_tokens,
                    "temperature": backend.temperature,
                    "synthesis_timeout": backend.synthesis_timeout,
                    "cc_allowed_tools": backend.cc_allowed_tools,
                }
                for backend_id, backend in self.llm_backends.items()
            },
            "fixed_tools": {
                tool_name: {
                    "command": tool.command,
                    "description": tool.description,
                    "timeout_hours": tool.timeout_hours,
                    "parameters": {
                        parameter.name: {
                            "type": parameter.type,
                            "description": parameter.description,
                            "required": parameter.required,
                            "cli_flag": parameter.cli_flag,
                        }
                        for parameter in tool.parameters
                    },
                    "phase_markers": list(tool.phase_markers),
                    "artifacts": list(tool.artifacts),
                }
                for tool_name, tool in self.fixed_tools.items()
            },
            "nodes": {
                node_id: {
                    "router_host": node.router_host,
                    "router_port": node.router_port,
                    "use_tls": node.use_tls,
                    "auth_token": node.auth_token,
                    "llm_model": node.llm_model,
                    "llm_backend": node.llm_backend,
                    "router_deep_backend": node.router_deep_backend,
                    "router_deep_enabled": node.router_deep_enabled,
                    "fold_backend": node.fold_backend,
                    "system_prompt": node.system_prompt,
                    "tools": node.tools,
                    **(
                        {"isolation": node.isolation.to_dict()}
                        if node.isolation is not None
                        else {}
                    ),
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
        access=backend.access,
        model=backend.default_model,
        max_tokens=backend.max_tokens,
        temperature=backend.temperature,
        synthesis_timeout=backend.synthesis_timeout,
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
        chat_template_kwargs=backend.chat_template_kwargs,
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
        codex_env=backend.codex_env,
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
