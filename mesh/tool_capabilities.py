"""Tool capability catalog and the single runtime authorization guard.

Phase 2A of the per-agent isolation plan.  Two things live here and nowhere
else:

1. :data:`TOOL_CAPABILITIES` — the auditable classification of every tool the
   mesh can execute, whether it comes from the global registry, Router V2's
   static list, or ``AgentNode``'s special dispatch.
2. :func:`guard_tool` — the one function every execution choke point calls
   before dispatching.  Duplicating the check per choke point is how a bypass
   gets introduced, so there is exactly one implementation.

Classification rules
--------------------

``LOCAL``
    Touches only local state (files, memory, digests, dossiers, shell).  The
    boundary for these is filesystem containment (Phase 2B), not this guard.
``EXTERNAL_NETWORK``
    Reaches a third-party data plane the agent controls the content of — Exa,
    Gmail, arXiv, the notes server, a browser, a store front.
``MESH_CONTROL``
    The agent's own control plane.  Never denied: cutting ``send_message`` or
    ``send_report`` does not contain an agent, it only stops it from reporting
    what it did.
``CREDENTIAL``
    Reads or rewrites host credential material (OAuth tokens, cookie jars,
    API-key stores, ``~/.claude/.credentials.json``).  Always requires an
    explicit name in ``allowed_credential_tools``, even when network is
    allowed.

A tool may carry several capabilities and must satisfy all of them.  The
model/router transport itself is *not* a tool capability: per the plan, a
trusted LLM driver reaching its provider is control-plane traffic, so tools
that merely invoke the configured model (``style_filter``, ``math_thinking``,
``mesh_qwen``, ``recursive_harness``) are not classified
``EXTERNAL_NETWORK``.  They are already reachable through the agent's own
inference path, so denying them would buy no containment.

Unclassified tools default to ``{LOCAL}``, which is the safe default, and
``tests/test_tool_capabilities.py`` fails the build when a new tool has no
entry here — the default must never become the silent path for a network tool.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from .isolation import Authorization, IsolationPolicy, ToolCapability, authorize_tool

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tools import ToolRegistry

__all__ = [
    "TOOL_CAPABILITIES",
    "DEFAULT_CAPABILITIES",
    "capabilities_for",
    "authorize",
    "guard_tool",
    "refusal_text",
    "filter_tool_names",
    "known_tool_names",
    "unclassified_tool_names",
]

_LOCAL = frozenset({ToolCapability.LOCAL})
_NET = frozenset({ToolCapability.EXTERNAL_NETWORK})
_CONTROL = frozenset({ToolCapability.MESH_CONTROL})
_CRED = frozenset({ToolCapability.CREDENTIAL})
_NET_CRED = frozenset({ToolCapability.EXTERNAL_NETWORK, ToolCapability.CREDENTIAL})
_LOCAL_CRED = frozenset({ToolCapability.LOCAL, ToolCapability.CREDENTIAL})
_LOCAL_CONTROL = frozenset({ToolCapability.LOCAL, ToolCapability.MESH_CONTROL})

#: The safe default for a tool with no catalog entry.
DEFAULT_CAPABILITIES: frozenset[ToolCapability] = _LOCAL


def _spread(names: Iterable[str], caps: frozenset[ToolCapability]) -> dict:
    return {name: caps for name in names}


TOOL_CAPABILITIES: dict[str, frozenset[ToolCapability]] = {
    # ── Mesh control plane ────────────────────────────────────────────
    # Never denied by authorize_tool(); listed so the coverage test can prove
    # the decision was deliberate rather than a default.
    **_spread(
        (
            "send_message",
            "send_report",
            "sleep",
            "agent_shutdown",
            "mesh_list",
            "mesh_status",
            "agent_status",
            "channel_list",
            "channel_members",
            "tool_help",
            "worker_launch",
            "worker_list",
            "worker_status",
            "worker_cancel",
            "worker_stop",
            "schedule_wake",
            "schedule_list",
            "schedule_cancel",
            "conversation_notes_get",
            "conversation_notes_set",
            "todo_add",
            "todo_list",
            "todo_remove",
            "todo_reorder",
            "todo_set_section_order",
            "todo_toggle",
            "todo_update",
        ),
        _CONTROL,
    ),
    # attach_file reads a local path and delivers it over the mesh: both.
    "attach_file": _LOCAL_CONTROL,
    # autonomous_controller_run drives the owning agent's own controller.
    "autonomous_controller_run": _LOCAL_CONTROL,

    # ── External network: research / web ──────────────────────────────
    **_spread(
        (
            "exa_search",
            "exa_fetch_full",
            "extract_url",
            "literature_search",
            "literature_fulltext",
            "arxiv_search",
            "arxiv_get",
            "arxiv_fulltext",
            "pubmed_search",
            "pubmed_get",
            "pubmed_related",
            "pubmed_fulltext",
            "openalex_search",
            "solicitation_scout",
            "synthetic_quota",
            # boox_upload pushes a file to a device over HTTP — egress.
            "boox_upload",
        ),
        _NET,
    ),
    # ── External network: notes server (HTTP, not local files) ────────
    **_spread(
        (
            "notes_add",
            "notes_delete",
            "notes_edit",
            "notes_get",
            "notes_list",
            "notes_read",
            "notes_search",
        ),
        _NET,
    ),
    # ── External network: browser automation ──────────────────────────
    **_spread(
        (
            "browser_back",
            "browser_click",
            "browser_fill",
            "browser_get_url",
            "browser_goto",
            "browser_press",
            "browser_read_text",
            "browser_select",
            "browser_session_close",
            "browser_session_open",
            "browser_session_status",
            "browser_snapshot_controls",
            "browser_type",
        ),
        _NET,
    ),
    # ── External network + credential: Google (OAuth token files) ─────
    **_spread(
        (
            "gmail_create_draft",
            "gmail_draft_reply",
            "gmail_get_email",
            "gmail_list_from_date",
            "gmail_list_recent",
            "gmail_list_unread",
            "gmail_reply_to",
            "gmail_search_emails",
            "gmail_send_message",
            "calendar_create_event",
            "calendar_delete_event",
            "calendar_list_on_date",
        ),
        _NET_CRED,
    ),
    # ── External network + credential: Plaid (API keys / access tokens) ─
    **_spread(
        (
            "plaid_accounts",
            "plaid_link_start",
            "plaid_link_status",
            "plaid_sync",
            "plaid_transactions",
            "plaid_unlink",
        ),
        _NET_CRED,
    ),
    # ── External network + credential: Canvas (~/.mesh/canvas_token.json) ─
    **_spread(
        (
            "canvas_auth_status",
            "canvas_create_assignment",
            "canvas_create_module",
            "canvas_create_page",
            "canvas_get_analytics",
            "canvas_get_grades",
            "canvas_get_page",
            "canvas_get_student",
            "canvas_grade_submission",
            "canvas_list_announcements",
            "canvas_list_assignments",
            "canvas_list_courses",
            "canvas_list_module_items",
            "canvas_list_modules",
            "canvas_list_pages",
            "canvas_list_quizzes",
            "canvas_list_students",
            "canvas_list_submissions",
            "canvas_post_announcement",
            "canvas_update_page",
            "canvas_upload_file",
        ),
        _NET_CRED,
    ),
    # ── Credential-bearing ────────────────────────────────────────────
    # claude_code_usage reads and rewrites ~/.claude/.credentials.json and
    # queries Anthropic's usage endpoint.
    "claude_code_usage": _NET_CRED,
    # Gmail account selection resolves and switches credential/token files.
    **_spread(
        ("account_get_current", "account_list", "account_set_current"),
        _CRED,
    ),
    # Starting a Claude Code session consumes host Claude credentials; the
    # plan forbids isolated agents from starting one until cc_session_manager
    # accepts a policy.  The interaction tools cannot create a session.
    "cc_start_session": _LOCAL_CRED,
    **_spread(
        ("cc_get_screen", "cc_send_input", "cc_stop_session"),
        _LOCAL,
    ),

    # ── Local: files, shell, harness ──────────────────────────────────
    **_spread(
        (
            "file_create",
            "file_diff",
            "file_edit",
            "file_read",
            "file_write",
            "write_lines",
            "list_dir",
            "grep",
            "get_context",
            "count_words",
            "token_count",
            "bash_exec",
            "get_working_directory",
            "set_working_directory",
            # Harness worker tools (mesh/harness/tools/*): registered into the
            # same global registry the moment mesh.harness is imported.
            "shell",
            "apply_patch",
            "find_files",
            "phase_complete",
            "current_time",
            "echo",
            "scratchpad_read",
            # Model-invoking tools: control-plane transport, see module docs.
            "style_filter",
            "math_thinking",
            "mesh_qwen",
            "recursive_harness",
            "harness_start_session",
            "harness_send_input",
            "harness_get_status",
            "harness_stop_session",
            "skill_draft",
        ),
        _LOCAL,
    ),
    # security_scan calls an OpenAI-compatible HTTP endpoint selected by its
    # ``api`` argument.  Even though the default is loopback, the tool can make
    # arbitrary outbound requests and must obey the external-network policy.
    "security_scan": _NET,
    # ── Local: memory / digest / essay / dossier / entity state ───────
    **_spread(
        (
            "remember",
            "memory_add",
            "memory_delete",
            "memory_edit",
            "memory_get",
            "memory_list",
            "memory_search",
            "history_search",
            "digest_edit",
            "digest_get",
            "essay_edit",
            "essay_get",
            "essay_list",
            "dossier_check_budget",
            "dossier_edit",
            "dossier_read",
            "dossier_spend_budget",
            "dossier_write_report",
            "entity_backfill",
            "entity_create",
            "entity_edit",
            "entity_group_create",
            "entity_group_member_add",
            "entity_group_member_remove",
            "entity_link_correct",
            "entity_merge",
            "map_create",
            "map_edit",
            "map_get",
            "map_list",
            "map_review",
            "set_project_context",
            "personality_get",
            "personality_set",
        ),
        _LOCAL,
    ),
}


def capabilities_for(
    tool_name: str,
    registry: "ToolRegistry | None" = None,
) -> frozenset[ToolCapability]:
    """Capabilities for ``tool_name``.

    The catalog is authoritative.  A tool registered with an explicit
    ``capabilities=`` argument that is not in the catalog falls back to its
    :class:`~mesh.tools.ToolDefinition`, so a plugin can classify itself.
    Anything still unknown is ``{LOCAL}``.
    """
    caps = TOOL_CAPABILITIES.get(tool_name)
    if caps is not None:
        return caps
    if registry is not None:
        tool_def = registry.get(tool_name)
        declared = getattr(tool_def, "capabilities", None)
        if declared:
            return frozenset(declared)
    return DEFAULT_CAPABILITIES


def authorize(
    policy: "IsolationPolicy | None",
    tool_name: str,
    registry: "ToolRegistry | None" = None,
) -> Authorization:
    """Authorize ``tool_name`` under ``policy`` using the catalog."""
    if policy is None or not policy.enabled:
        # Legacy fast path: no catalog lookup, no allocation, no work.
        return Authorization(allowed=True, tool_name=tool_name)
    return authorize_tool(policy, tool_name, capabilities_for(tool_name, registry))


def refusal_text(auth: Authorization) -> str:
    """The stable, string-shaped refusal handed back to a tool caller.

    Names the tool, the denied capability and the policy source and nothing
    else — no credential paths, no roots the caller does not already know.
    Prefixed ``Error: `` so it flows through the existing error plumbing
    (``_track`` records it as a rejection, harness loops surface it).
    """
    capability = auth.denied_capability.value if auth.denied_capability else "unknown"
    return (
        f"Error: isolation_denied — tool '{auth.tool_name}' requires the "
        f"'{capability}' capability, which this agent's isolation policy "
        f"({auth.policy_source or 'unknown'}) does not grant. {auth.reason}"
    ).strip()


def guard_tool(
    policy: "IsolationPolicy | None",
    tool_name: str,
    registry: "ToolRegistry | None" = None,
) -> str | None:
    """The single execution-time guard.

    Returns ``None`` when the call may proceed and a refusal string when it
    may not.  Every choke point calls exactly this; a disabled or absent
    policy returns ``None`` after one attribute check, which is what keeps the
    legacy path a fast path.
    """
    if policy is None or not policy.enabled:
        return None
    auth = authorize(policy, tool_name, registry)
    if auth.allowed:
        return None
    return refusal_text(auth)


def filter_tool_names(
    names: Iterable[str],
    policy: "IsolationPolicy | None",
    registry: "ToolRegistry | None" = None,
) -> list[str]:
    """Offer-time filter: drop names the policy would refuse to execute.

    A disabled policy returns the names unchanged (same order, same objects),
    so no agent that is not isolated sees a different tool list.
    """
    name_list = list(names)
    if policy is None or not policy.enabled:
        return name_list
    return [n for n in name_list if authorize(policy, n, registry).allowed]


def known_tool_names() -> set[str]:
    """Every tool name Phase 2A must have a classification for.

    Union of the global registry, Router V2's static list and its gated
    extensions, and ``AgentNode``'s special-dispatch names.  Imported lazily
    because both modules import this one.

    The harness tool package is imported explicitly: it registers ``shell``,
    ``apply_patch``, ``find_files`` and ``phase_complete`` into the *same*
    global registry, so without this the answer would depend on whether some
    earlier import happened to pull ``mesh.harness`` in — exactly the kind of
    order-dependent coverage gap this function exists to close.
    """
    import importlib
    import pkgutil

    from .tools import get_registry

    from . import tool_implementations  # noqa: F401  (registers ~180 tools)
    from .harness import tools as _harness_tools

    for module in pkgutil.iter_modules(_harness_tools.__path__):
        importlib.import_module(f"{_harness_tools.__name__}.{module.name}")

    names: set[str] = set(get_registry().list_names())

    from . import router_v2

    names |= set(router_v2.ROUTER_TOOL_NAMES)
    names |= set(router_v2.WORKER_ROUTER_TOOLS)
    names |= set(router_v2.MANAGED_WORKER_ROUTER_TOOLS)
    names |= set(router_v2.FIXED_TOOL_ROUTER_TOOLS)
    names |= set(router_v2.CC_INTERACTIVE_TOOLS)
    names |= set(router_v2.HARNESS_SESSION_INTERACTIVE_TOOLS)

    from .agent_node import AgentNode

    names |= set(AgentNode._TODO_TOOL_NAMES)
    names |= set(AgentNode._CONVERSATION_NOTES_TOOL_NAMES)
    names |= set(AgentNode._ENTITY_TOOL_NAMES)
    names |= set(AgentNode._CURATION_REUSED_MUTATION_NAMES)
    names |= {
        "send_message",
        "send_report",
        "attach_file",
        "channel_list",
        "channel_members",
        "schedule_wake",
        "schedule_list",
        "schedule_cancel",
        "agent_shutdown",
        "mesh_status",
        "agent_status",
        "sleep",
        "worker_stop",
    }
    return names


def unclassified_tool_names() -> set[str]:
    """Names that would silently fall back to ``{LOCAL}``.  Must stay empty."""
    return known_tool_names() - set(TOOL_CAPABILITIES)
