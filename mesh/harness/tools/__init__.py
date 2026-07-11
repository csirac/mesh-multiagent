"""
Harness-specific tool implementations.

These mirror the Codex tool surface:
- apply_patch: V4A envelope format patch applier with fuzzy matching
- shell: Persistent shell sessions with optional PTY, or one-shot subprocess
- list_dir: Tree-style directory listing
- file_read: Read files with required start_line + num_lines (also accepts end_line)

Import this module to register all harness tools with the global ToolRegistry.
"""

from . import apply_patch  # noqa: F401
from . import unified_exec  # noqa: F401
from . import list_dir  # noqa: F401
from . import file_read  # noqa: F401
from . import file_edit  # noqa: F401
from . import phase_complete  # noqa: F401
from . import grep  # noqa: F401
from . import find_files  # noqa: F401

HARNESS_TOOLS = [
    "apply_patch",
    "shell",
    "list_dir",
    "file_read",
    "file_edit",
    "phase_complete",
    "grep",
    "find_files",
]

PLANNER_TOOLS = [
    "file_read",
    "list_dir",
    "grep",
    "find_files",
]

MESH_READ_ONLY_TOOLS = frozenset({
    "gmail_search_emails",
    "gmail_list_from_date",
    "gmail_get_email",
    "exa_search",
    "exa_fetch_full",
    "notes_search",
    "notes_list",
    "notes_get",
    "notes_read",
    "arxiv_search",
    "arxiv_get",
    "arxiv_fulltext",
    "pubmed_search",
    "pubmed_get",
    "pubmed_fulltext",
    "pubmed_related",
    "literature_search",
    "literature_fulltext",
    "extract_url",
    "calendar_list_on_date",
    "memory_search",
    "memory_get",
    "memory_list",
    "history_search",
    "map_get",
    "map_list",
    "current_time",
    "tool_help",
    "mesh_list",
    "mesh_status",
    "agent_status",
    "channel_list",
    "channel_members",
    "plaid_link_status",
    "plaid_accounts",
    "plaid_transactions",
    "synthetic_quota",
    "claude_code_usage",
    "account_get_current",
    "account_list",
    "browser_session_status",
    "browser_get_url",
    "browser_snapshot_controls",
    "browser_read_text",
    "schedule_list",
    "personality_get",
})
