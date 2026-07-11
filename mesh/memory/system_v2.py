"""
Memory system v2 — project-oriented agent memory.

Components:
1. Maps — living project documents (separate `project_maps` table)
2. Log entries — episodic records (`memories` table + `project` column)
3. Retrieval — on-demand semantic search over log entries (Phase 5)
4. Representative memories — FLMI-selected from full pool (retained from v1)

Phase 1 scope: Map store, CRUD, initialization scan, agent-facing tools,
active project persistence across restarts.
"""

import asyncio
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .embeddings import EmbeddingClient
from .selection import (
    _build_sim_matrix,
    compute_withholding_costs,
    cosine_sim,
    lazy_greedy_fl,
    select_active_set,
    try_swap,
)
from .store import MemoryEntry, MemoryStore, _deserialize_embedding

logger = logging.getLogger(__name__)


def _reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using reciprocal rank fusion (RRF).

    Each ranking is a list of (id, score) sorted by score descending.
    Returns merged (id, rrf_score) sorted by rrf_score descending.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (entry_id, _score) in enumerate(ranking):
            scores[entry_id] = scores.get(entry_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _turn_ts_iso(turn) -> str:
    """Best-effort extraction of a turn's timestamp as an ISO 8601 string."""
    ts = getattr(turn, "timestamp", "")
    if isinstance(ts, datetime):
        return ts.isoformat()
    if isinstance(ts, str):
        return ts
    return ""


def _extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag> and </tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# Scan pipeline: files to prioritize for map initialization
_PRIORITY_GLOBS = {
    "docs": ["README*", "CONTRIBUTING*", "docs/*"],
    "config": [
        "*.yaml", "*.yml", "*.toml", "*.json", "Makefile",
        "pyproject.toml", "setup.py", "setup.cfg",
    ],
    "entry_points": [
        "main.py", "app.py", "__main__.py", "router*.py",
        "server*.py", "cli.py",
    ],
    "init_files": ["*/__init__.py"],
    # Non-code projects (LaTeX, notes, org-mode, etc.)
    "text": [
        "*.tex", "*.md", "*.txt", "*.org", "*.rst",
        "*.bib", "*.sty",
    ],
}

# ── Conversation summary prompt ──────────────────────────────────

CONVERSATION_SUMMARY_PROMPT = """\
You are updating a running conversation summary for an AI agent.

The summary captures the conversational thread — what was discussed, what was
requested, what was decided — so the agent can pick up naturally after context
is compressed. Focus on the dialogue flow: who asked what, what the outcomes
were, and what is pending.

(Reflections and project maps handle lessons-learned and project state
separately. This summary only needs to capture the conversation itself.)

## Rules
- Merge the existing summary with the new turns into a single coherent narrative.
- Organize by topic with timestamps and status: [ACTIVE], [COMPLETED], [PENDING].
- Compress completed topics from the existing summary to 1-2 sentences.
- Give the most recent / active topics the most detail.
- Preserve specifics: file paths, names, commands, decisions, error messages.
- Target length: 600-1200 words. Never exceed 1500 words.

{existing_summary_section}

[New turns to incorporate]
{new_turns}

Produce the updated summary. Output ONLY the summary text — no preamble."""

# Max tokens for conversation summary (soft cap, truncated if exceeded)
SUMMARY_MAX_TOKENS = 3000

# Map synthesis prompt
MAP_SYNTHESIS_PROMPT = """You are building a project map for an AI agent's memory system.

Given the directory structure and source files below, create a project knowledge map
with the following sections:

# Project: {project_name}

## Architecture
High-level description: components, relationships, data flow.
Key file paths and module boundaries.

## Components
Bulleted list of major components with 1-2 sentence descriptions.
Include key file:line references where the agent should start looking.

## Current State
What's deployed, what's in progress, what's broken.

## Open Questions
Active design decisions, unresolved issues.

## Key Decisions
Important decisions made, with rationale.

Target length: 800-1,500 words. Each major component gets 2-3 sentences.
Point to entry files and key modules — don't list every filename.

If you need additional files to produce a complete map, include a block:
<request_files>path1, path2, ...</request_files>

Otherwise, output ONLY the map document (starting with # Project:).

<directory_structure>
{tree_output}
</directory_structure>

<source_files>
{file_contents}
</source_files>"""


# Enhanced reflection prompt for window drop segments (adds format enforcement + project names)
WINDOW_DROP_REFLECTION_PROMPT = """You just completed a segment of work. Reflect deeply on what happened.

<task_trigger>
{trigger}
</task_trigger>

<trace>
{trace}
</trace>

Produce your response in exactly this format:

<reflection>
2-3 paragraphs. What did you do? What worked? What didn't?
Did you have a plan? Did it go wrong? What specific strategies led to the
outcome? What would you do differently?
</reflection>

<summary>
One paragraph distilling the reflection. What happened + key lesson.
</summary>

<tags>
comma-separated domain tags (e.g., nginx, benchmark, mesh-routing)
</tags>

<outcome_label>
success | partial | failure
</outcome_label>

<retrieval_key>
In 1-2 sentences, describe what this task or conversation was about.
Be specific — name technologies, systems, files, and concepts involved.
Focus on the primary topic or task. If the session covered multiple topics,
describe the most significant one.
This will be used as a search key, so include terms a future query would match.
</retrieval_key>

<project>
Which project does this work belong to? Your current projects:
{known_project_names}
Use one of these names exactly. If it spans multiple projects, name the
primary one. If it doesn't match any existing project, suggest a short
hyphenated name for a new project (e.g., "data-analysis", "llm-eval").
</project>

IMPORTANT: You MUST close every XML tag. Each opening <tag> MUST have
a matching </tag>. Unclosed tags will cause parsing failures.

Example of correct format:

<reflection>
Your 2-3 paragraph reflection here.
</reflection>

<summary>
One paragraph summary here.
</summary>

<tags>
mesh-routing, debugging, asyncio
</tags>

<outcome_label>
success
</outcome_label>

<retrieval_key>
Descriptive search key here with specific technologies and concepts.
</retrieval_key>

<project>
mesh-system
</project>"""

# Canonical project map section schema (shared by all map prompts)
_MAP_SECTIONS_DOC = """\
STANDARD SECTIONS (create if missing):
- ## Summary — 2-3 sentences: what this project is, current phase.
- ## Goals — what we're trying to accomplish, success criteria, guiding \
constraints. This is the MOST IMPORTANT section. When an agent must make \
a judgment call on an underspecified task, Goals is the anchor. Write it \
so a reader who knows nothing else can evaluate whether a decision aligns \
with the project's intent.
- ## Architecture — system structure, data flow, key abstractions.
- ## Key Files — navigational map with file paths, grouped by subsystem, \
with relationships noted. Not a flat list — show how files connect.
- ## Key Decisions — decisions made, rationale, dates.
- ## Current State — what's active, completed, blocked, next.
- ## Open Issues — unresolved problems, open questions.
- ## Glossary — abbreviations, project-specific vocabulary."""

# Map curation prompt (passive — at window drop)
MAP_CURATION_PROMPT = """You are maintaining a project knowledge map. Below is the current map \
and a conversation that just dropped from the rolling window.

Your job: extract new information from the conversation and output ONLY \
the map sections that need updating. Sections you don't mention stay unchanged.

PRIORITY: The conversation is split into USER MESSAGES and AGENT RESPONSES. \
User messages are the PRIMARY source — they define vocabulary, make decisions, \
explain concepts, and set direction. Read the user messages FIRST.

WHAT TO EXTRACT:

1. **Vocabulary** — Every abbreviation or term the user defines or uses. \
   "baseline is around 5 to 8%" defines "baseline" AND gives its metric. \
   "forced controller sequence is 12 to 15%" defines FC with its metric. \
   These definitions are LOST if you don't capture them. Add to Glossary.

2. **Goals and intent** — Any statement about what the project aims to \
   achieve, success criteria, or constraints. These are critical — agents \
   lose sight of project intent over time, and this section anchors them.

3. **Results** — Every metric, percentage, comparison. Preserve EXACT numbers. \
   Pay special attention to user comparisons between conditions.

4. **Methods** — Every algorithm, architecture, technique mentioned.

5. **Decisions** — Why something was chosen or abandoned.

6. **Architecture** — New files, modules, data flow changes.

7. **Current state** — What's running, what's next, what's blocked.

OUTPUT FORMAT:

If nothing relevant, respond exactly: "No updates needed."

Otherwise, output one or more <section> blocks. Each block contains the \
COMPLETE content for that section (it replaces the existing section). \
Only include sections that actually changed.

<section name="## Glossary">
...complete glossary content with new entries added...
</section>

<section name="## Key Decisions">
...complete decisions section with new entries added...
</section>

You can also ADD entries to a section without replacing it entirely:

<append_to name="## Glossary">
- **BL (Baseline):** Description here.
</append_to>

<append_to name="## Open Issues">
- **Issue title** — description.
</append_to>

Use <append_to> when adding new entries to a section. \
Use <section> when the section needs restructuring or significant changes.

""" + _MAP_SECTIONS_DOC + """

SELF-CHECK before outputting:
- Did you capture EVERY abbreviation defined or used in the conversation?
- Did you capture EVERY percentage/metric?
- Did you capture EVERY algorithm or technique name?
- Did you capture any goals, success criteria, or intent statements?
If you missed any, go back and add them.

Conversation (more recent) wins over existing map content on contradictions.

TOOLS (rarely needed):
- <file_read>path/to/file</file_read> — read a file
- <bash_exec>command</bash_exec> — run a read-only shell command
Only use tools to resolve contradictions that need external evidence.

<current_map>
{current_map}
</current_map>

<dropped_conversation>
{raw_conversation}
</dropped_conversation>"""

# New project initialization prompt (when reflection names an unknown project)
NEW_PROJECT_MAP_PROMPT = """A new project has been identified: "{project_name}"

Based on the following work summary, create an initial project map.

""" + _MAP_SECTIONS_DOC + """

Keep it minimal — this is a starting point that will be refined as \
more work is done on this project. For Goals, infer the project's \
purpose and success criteria from the summary. Even a rough Goals \
section is better than none — it will be refined as more context arrives.

Populate every section with at least a placeholder if you have any \
signal at all. Empty sections are fine — they'll be filled by the \
integration pipeline.

<work_summary>
{summary}
</work_summary>

Output the map document starting with # Project: {project_name}"""

# Entry-driven map integration prompt (runs after V3 formation)
ENTRY_INTEGRATION_PROMPT = """\
You are maintaining a project knowledge map. Below is the current map \
and a set of new memory entries that were just created from recent work \
on this project.

Your job: integrate the new information into the map using targeted edits.

WHAT TO UPDATE:

1. **Summary** — adjust if the project's phase or scope has shifted
2. **Goals** — update if entries reveal new objectives, refined success \
criteria, or changed constraints. This is the most important section — \
it anchors agent judgment calls on underspecified tasks.
3. **Architecture** — structural changes, new components, modified data flow
4. **Key Files** — new files mentioned in entries, changed paths, relationships
5. **Key Decisions** — decisions made, rationale, alternatives considered
6. **Current State** — what's active, what completed, what's blocked
7. **Open Issues** — new issues raised, resolved issues to remove
8. **Glossary** — new terms, abbreviations defined

OUTPUT FORMAT:

If the entries contain nothing map-worthy, respond exactly: "No updates needed."

Otherwise, output one or more <section> or <append_to> blocks:

<append_to name="## Open Issues">
- **Issue title** — description (from entry date)
</append_to>

<section name="## Current State">
...complete replacement for this section...
</section>

Use <append_to> to add new items to a section.
Use <section> to rewrite a section that needs restructuring.

""" + _MAP_SECTIONS_DOC + """

RULES:
- Preserve existing content unless contradicted by new entries.
- New entries win on contradictions (they are more recent).
- Do not remove items from Open Issues unless an entry explicitly resolves them.
- Keep entries concise — one line per item, no paragraphs.

<current_map>
{current_map}
</current_map>

<new_entries>
{entries_text}
</new_entries>"""

# Standalone consistency audit prompt
MAP_AUDIT_PROMPT = """You are auditing a project knowledge map for internal consistency.

Read the map below carefully. Identify any contradictions, stale references,
or inconsistencies between sections. For each issue found, describe it in
one sentence.

If the map is internally consistent, respond exactly: "No issues found."

Otherwise, output each inconsistency on its own line, prefixed with "- ".

<map>
{map_content}
</map>"""

# Map review prompt (intensive reconciliation against filesystem)
MAP_REVIEW_PROMPT = """You are performing a deep review of a project knowledge map, reconciling it
against the actual current state of the project on disk.

Your goal: bring the map fully up to date. The map may have drifted — files renamed,
configs changed, components added or removed, status outdated. Check everything.

PROJECT DIRECTORY: {project_dir}

TOOLS — you MUST use these to verify every claim in the map:

- <file_read>path/to/file</file_read> — read a file's contents
- <bash_exec>command</bash_exec> — run a shell command (ls, find, head, wc, grep, etc.)

IMPORTANT: The orientation hint below (if present) shows only the directory tree —
it does NOT show file contents. You MUST read the actual source files to verify
specific values like thresholds, channel counts, architecture parameters, and
dependency lists. Do NOT guess these from file names alone.

PROCEDURE:

1. START by reading the key source files mentioned in the map. For each file listed,
   use <file_read> to read it and verify the map's description is accurate.
   Also read requirements.txt, config files, and any files the map references.

2. For EACH claim in the map:
   - Verify the file/directory exists at the stated path
   - Read the file to confirm specific values (thresholds, dimensions, parameters)
   - Check if configs, architecture, or status have changed

3. Look for NEW components on disk that the map doesn't mention.
   Explore directories the map is silent about.

4. Check for internal contradictions — sections that disagree with each other.

5. For each discrepancy:
   - Check recent notes files, git log, and any relevant configs to resolve it.
   - Use your best judgment. Do NOT ask the user — resolve ambiguities yourself
     by reading files, checking notes, and inspecting history.
   - If something was deleted, remove it from the map.
   - If something was added, add it to the map with a description based on reading the files.

You MUST make tool calls to read files BEFORE producing the updated map.
Do NOT produce <updated_map> until you have verified the map's claims by
reading the actual source files.

When you have finished exploring and are ready to produce the updated map,
output this block:

<updated_map>
The COMPLETE updated map document starting with # Project: ...
Include ALL sections. Do not omit unchanged sections.
</updated_map>

Map guidelines: the map is a roadmap, not an encyclopedia. Point to where the
relevant code, config, and decisions live. Don't embed values that change
frequently (iteration counts, specific metric values, in-progress percentages).

<current_map>
{current_map}
</current_map>

{orientation}

{recent_context}"""


def _parse_tool_calls(text: str) -> list[tuple[str, str]]:
    """Parse tool call tags from LLM response.

    Returns list of (tool_name, argument) tuples.
    Only recognizes file_read and bash_exec.
    """
    calls = []
    for tag in ("file_read", "bash_exec"):
        for match in re.finditer(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL):
            calls.append((tag, match.group(1).strip()))
    return calls


async def _execute_tool_call(
    tool_name: str, arg: str, project_dir: str | None = None,
) -> str:
    """Execute a curation tool call (file_read or bash_exec).

    Returns the tool result as a string. Read-only operations only.
    project_dir is used to resolve relative file_read paths.
    """
    if tool_name == "file_read":
        try:
            path = MemorySystemV2._resolve_home(arg)
            # Resolve relative paths against project_dir
            if not os.path.isabs(path) and project_dir:
                path = os.path.join(project_dir, path)
            with open(path, "r", errors="replace") as f:
                content = f.read(100_000)  # 100KB cap
            return content
        except Exception as e:
            return f"Error reading {arg}: {e}"
    elif tool_name == "bash_exec":
        # Read-only shell command with timeout
        try:
            proc = await asyncio.create_subprocess_shell(
                arg,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_dir,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=15,
            )
            result = stdout.decode(errors="replace")[:50_000]
            if stderr:
                result += "\n" + stderr.decode(errors="replace")[:10_000]
            return result
        except asyncio.TimeoutError:
            return f"Command timed out: {arg}"
        except Exception as e:
            return f"Error executing command: {e}"
    return f"Unknown tool: {tool_name}"


def _apply_section_updates(current_map: str, llm_response: str) -> str | None:
    """Parse <section> and <append_to> tags from LLM response and apply to map.

    Returns updated map string, or None if no valid updates found.
    """
    replacements = {}  # section_name -> new content
    appends = {}       # section_name -> text to append

    for match in re.finditer(
        r'<section\s+name="(##\s+[^"]+)">(.*?)</section>',
        llm_response, re.DOTALL,
    ):
        replacements[match.group(1).strip()] = match.group(2).strip()

    for match in re.finditer(
        r'<append_to\s+name="(##\s+[^"]+)">(.*?)</append_to>',
        llm_response, re.DOTALL,
    ):
        name = match.group(1).strip()
        appends.setdefault(name, []).append(match.group(2).strip())

    if not replacements and not appends:
        return None

    result = current_map

    for section_name, new_content in replacements.items():
        # Find section in map and replace it
        pattern = re.compile(
            r'(^' + re.escape(section_name) + r')\s*\n(.*?)(?=\n## |\Z)',
            re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(result)
        if match:
            result = result[:match.start()] + section_name + "\n" + new_content + result[match.end():]
        else:
            # Section doesn't exist — append it
            result = result.rstrip() + "\n\n" + section_name + "\n" + new_content

    for section_name, texts in appends.items():
        append_text = "\n".join(texts)
        # Find end of section
        pattern = re.compile(
            r'(^' + re.escape(section_name) + r'\s*\n.*?)(?=\n## |\Z)',
            re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(result)
        if match:
            result = result[:match.end()] + "\n" + append_text + result[match.end():]
        else:
            # Section doesn't exist — create it
            result = result.rstrip() + "\n\n" + section_name + "\n" + append_text

    return result


@dataclass
class TocEntry:
    """One TOC entry: enough to display + decide whether to fetch."""
    id: str
    retrieval_key: str
    project: str
    tags: list[str] = field(default_factory=list)
    already_in_context: bool = False
    truncated_in_context: bool = False
    score: float = 0.0


class MemorySystemV2:
    """Project-oriented agent memory system.

    Components:
    1. Maps — living project documents (separate `project_maps` table)
    2. Log entries — episodic records (`memories` table + `project` column)
    3. Retrieval — on-demand semantic search over log entries (Phase 5)
    4. Representative memories — FLMI-selected from full pool (retained from v1)
    """

    def __init__(
        self,
        nickname: str,
        llm_client,
        pool_max_entries: int = 1000,
        embedding_backend: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        recent_log_count: int = 4,
        retrieve_budget_tokens: int = 6000,
        retrieve_max_rounds: int = 2,
        reflection_min_tools: int = 3,
        retrieval_k: int = 5,
        trace_max_tokens: int = 2000,
        reflection_cooldown_secs: int = 300,
        reflection_min_discussion_turns: int = 4,
        reflection_min_discussion_chars: int = 1500,
        reflection_min_brainstorm_response_chars: int = 1500,
        reflection_max_brainstorm_tools: int = 2,
        curation_audit_max_tool_calls: int = 10,
        review_max_tool_calls: int = 30,
        # FLMI active set size (retained from v1)
        active_size: int = 30,
        # Memory Formation v3 (rev 6)
        formation_v3_enabled: bool = False,
        formation_v3_window_size: int = 60,
        formation_v3_overlap: int = 20,
        formation_v3_defer_tail: int = 10,
        formation_v3_model: str | None = None,
        formation_v3_parse_failure_fallback_threshold: int = 3,
        payload_max_chars: int = 6000,
        formation_llm_client=None,
        **kwargs,
    ):
        self._nickname = nickname
        self._payload_max_chars = payload_max_chars
        self._llm_client = llm_client
        self._formation_llm_client = formation_llm_client
        self._pool_max_entries = pool_max_entries
        self._recent_log_count = recent_log_count
        self._retrieve_budget_tokens = retrieve_budget_tokens
        self._retrieve_max_rounds = retrieve_max_rounds
        self._reflection_min_tools = reflection_min_tools
        self._retrieval_k = retrieval_k
        self._trace_max_tokens = trace_max_tokens
        self._reflection_cooldown_secs = reflection_cooldown_secs
        self._reflection_min_discussion_turns = reflection_min_discussion_turns
        self._reflection_min_discussion_chars = reflection_min_discussion_chars
        self._reflection_min_brainstorm_response_chars = reflection_min_brainstorm_response_chars
        self._reflection_max_brainstorm_tools = reflection_max_brainstorm_tools
        self._curation_audit_max_tool_calls = curation_audit_max_tool_calls
        self._review_max_tool_calls = review_max_tool_calls
        self._active_size = active_size
        self._last_reflection_time: float = 0.0

        self._store: MemoryStore | None = None
        self._embedder = EmbeddingClient(
            backend=embedding_backend,
            model=embedding_model,
        )

        # Active project state
        self._active_project: str | None = None
        self._active_project_dir: str | None = None  # set by set_project_context

        # In-memory state (pool + FLMI active set — retained from v1)
        self._pool: list[MemoryEntry] = []
        self._active_ids: set[str] = set()
        self._active_weights: dict[str, float] = {}
        self._active_f_s: float = 0.0

        # Personality cache
        self._personality_cache: str = ""

        # Query embedding cache
        self._query_embedding_cache: dict[int, np.ndarray] = {}

        # Map embedding cache (project_name -> embedding array)
        self._map_embedding_cache: dict[str, np.ndarray] = {}

        # Conversation summary (loaded from DB on init)
        self._conversation_summary: str = ""
        self._summary_messages_summarized: int = 0

        # ── Memory Formation v3 (rev 6) ──────────────────────────
        self._formation_v3_enabled: bool = formation_v3_enabled
        self._formation_v3_window_size: int = formation_v3_window_size
        self._formation_v3_overlap: int = formation_v3_overlap
        self._formation_v3_defer_tail: int = formation_v3_defer_tail
        self._formation_v3_model: str | None = formation_v3_model
        self._parse_failure_fallback_threshold: int = formation_v3_parse_failure_fallback_threshold
        # asyncio.Lock created lazily in initialize() (needs running loop)
        self._formation_lock: asyncio.Lock | None = None
        # Per-window failure counter, keyed by (cursor_idx, end_idx).
        self._parse_failure_count: dict[tuple[int, int], int] = {}
        # Optional callback fired after the cursor advances (success or fallback).
        # Used by AgentNode to reset its `_uncommitted_token_count` (§2.7.9).
        self._on_cursor_advance: callable | None = None

    @property
    def active_entries(self) -> list[MemoryEntry]:
        """Entries in the FLMI active set, preserving pool order."""
        return [e for e in self._pool if e.id in self._active_ids]

    async def initialize(self) -> None:
        """Open the store, load entries, restore active project."""
        from ..paths import MAPS_DIR
        os.makedirs(str(MAPS_DIR), exist_ok=True)
        self._store = MemoryStore(self._nickname)
        self._pool = self._store.load()
        self._personality_cache = self._store.get_personality()
        # Create the formation lock under the running loop.
        self._formation_lock = asyncio.Lock()

        # Restore active project from DB
        self._active_project = self._store.get_active_project()
        if self._active_project:
            self._active_project_dir = self._store.get_project_dir(
                self._active_project
            )
            logger.info(
                "Restored active project from DB: %s (dir=%s)",
                self._active_project, self._active_project_dir,
            )

        # Restore conversation summary from DB
        summary_row = self._store.get_summary()
        if summary_row:
            self._conversation_summary = summary_row["summary_text"]
            self._summary_messages_summarized = summary_row["messages_summarized"]
            logger.info(
                "Restored conversation summary: %d chars, %d messages covered",
                len(self._conversation_summary),
                self._summary_messages_summarized,
            )

        # FLMI active set (retained from v1 for representative block)
        self._reselect_active_set()
        self._prune_pool()

        logger.info(
            "MemorySystemV2 initialized for '%s': "
            "%d entries in pool, %d in active set, active_project=%s",
            self._nickname, len(self._pool), len(self._active_ids),
            self._active_project,
        )

        # Check for pending checkpoint from a previous crash
        replayed = await self.replay_checkpoint()
        if replayed:
            logger.info("Pending checkpoint replayed on startup")

        # One-time migration: ensure DB maps have files in central directory
        await self._migrate_maps_to_central()

    async def _migrate_maps_to_central(self) -> None:
        """Write DB-only maps to the central maps directory and backfill summaries."""
        from ..paths import MAPS_DIR
        maps_dir = str(MAPS_DIR)
        if not self._store:
            return
        for m in self._store.list_maps():
            name = m["project_name"]
            central = os.path.join(maps_dir, f"{name}.md")
            if os.path.exists(central):
                continue
            # Check old project_dir location
            old_dir = self._store.get_project_dir(name)
            if old_dir:
                old_path = os.path.join(old_dir, "PROJECT_MAP.md")
                if os.path.exists(old_path):
                    try:
                        with open(old_path, "r") as f:
                            content = f.read()
                        os.makedirs(maps_dir, exist_ok=True)
                        with open(central, "w") as f:
                            f.write(content)
                        logger.info("Migrated map '%s' from %s", name, old_path)
                    except Exception as e:
                        logger.warning("Map migration failed '%s': %s", name, e)
                    continue
            # DB-only map — write content to central if available
            row = self._store._conn.execute(
                "SELECT content FROM project_maps WHERE project_name = ?",
                (name,),
            ).fetchone()
            if row and row[0] and row[0].strip():
                try:
                    os.makedirs(maps_dir, exist_ok=True)
                    with open(central, "w") as f:
                        f.write(row[0])
                    logger.info("Wrote DB-only map '%s' to %s", name, central)
                except Exception as e:
                    logger.warning("DB map write failed '%s': %s", name, e)

        # Backfill summaries + embeddings for maps that have none
        for name, summary, emb_blob in self._store.list_map_embeddings():
            if summary and emb_blob:
                continue
            content = await self.get_map(name)
            if content:
                asyncio.create_task(
                    self._update_map_summary_and_embedding(name, content)
                )

    # ── Maps ──────────────────────────────────────────────────

    def _map_file_path(self, project_name: str) -> str:
        """Resolve the map file path for a project.

        Returns the central path: ~/.mesh/memory/maps/{project_name}.md.
        On first access, migrates from old {project_dir}/PROJECT_MAP.md if
        the central file doesn't exist but the old one does.
        """
        from ..paths import MAPS_DIR
        central = os.path.join(str(MAPS_DIR), f"{project_name}.md")
        if os.path.exists(central):
            return central
        # Backward-compat: check old project_dir location and migrate
        old_dir = None
        if project_name == self._active_project and self._active_project_dir:
            old_dir = self._active_project_dir
        if not old_dir and self._store:
            old_dir = self._store.get_project_dir(project_name)
        if old_dir:
            old_path = os.path.join(old_dir, "PROJECT_MAP.md")
            if os.path.exists(old_path):
                try:
                    os.makedirs(os.path.dirname(central), exist_ok=True)
                    with open(old_path, "r") as f:
                        content = f.read()
                    with open(central, "w") as f:
                        f.write(content)
                    logger.info(
                        "Migrated map '%s': %s → %s",
                        project_name, old_path, central,
                    )
                except Exception as e:
                    logger.warning("Map migration failed for '%s': %s", project_name, e)
                    return old_path
        return central

    async def get_map(self, project_name: str) -> str | None:
        """Get a project map's content from the central maps directory."""
        path = self._map_file_path(project_name)
        try:
            with open(path, "r") as f:
                content = f.read()
            logger.debug("Map '%s' loaded from %s: %d chars", project_name, path, len(content))
            return content
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error("Map '%s' read failed (%s): %s", project_name, path, e)
            return None

    def active_map_age_hours(self) -> float | None:
        """Hours since the active project map file was last modified, or None."""
        if not self._active_project:
            return None
        path = self._map_file_path(self._active_project)
        if not path or not os.path.exists(path):
            return None
        try:
            mtime = os.path.getmtime(path)
            age_seconds = time.time() - mtime
            return age_seconds / 3600
        except (OSError, TypeError):
            return None

    async def curate_active_map(
        self, raw_conversation: str, turn_count: int = 0,
    ) -> None:
        """Public interface for map curation from raw conversation text.

        Called by the router for staleness-based curation (no window drop).
        """
        await self._curate_active_map(raw_conversation, turn_count)

    async def list_maps(self) -> list[dict]:
        """List all project maps (metadata only, no content)."""
        maps = self._store.list_maps()
        for m in maps:
            fpath = self._map_file_path(m["project_name"])
            if os.path.exists(fpath):
                mtime = os.path.getmtime(fpath)
                m["updated_at"] = datetime.fromtimestamp(
                    mtime, tz=timezone.utc
                ).isoformat()
        return maps

    async def create_map(
        self, project_name: str, content: str,
        project_dir: str | None = None,
    ) -> bool:
        """Create a new project map in the central maps directory.

        Returns True on success, False on any failure.
        """
        import sqlite3
        file_path = self._map_file_path(project_name)
        if not content or not content.strip():
            logger.error(
                "Map '%s' create FAILED: refusing to write empty map to %s",
                project_name, file_path,
            )
            return False
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                f.write(content)
            logger.info("Map created: '%s' → %s (%d chars)", project_name, file_path, len(content))
        except Exception as e:
            logger.error("Map '%s' create FAILED (write error): %s", project_name, e)
            return False

        # Store project_dir in DB if provided (for working directory resolution)
        target_dir = project_dir or self._active_project_dir
        if not target_dir:
            stored_dir = self._store.get_project_dir(project_name)
            if stored_dir:
                target_dir = stored_dir

        map_id = uuid.uuid4().hex[:12]
        try:
            self._store.create_map(
                map_id, project_name, content, project_dir=target_dir,
            )
        except sqlite3.IntegrityError:
            self._store.update_map(project_name, content, project_dir=target_dir)

        asyncio.create_task(self._update_map_summary_and_embedding(project_name, content))
        return True

    async def update_map(self, project_name: str, content: str) -> bool:
        """Full content overwrite — writes map to the central maps directory.

        Returns True if the write succeeded AND was verified via read-back.
        Returns False on any failure.
        """
        path = self._map_file_path(project_name)

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
        except Exception as e:
            logger.error("Map '%s' update FAILED (write error): %s", project_name, e)
            return False

        # Read-back verification
        try:
            with open(path, "r") as f:
                persisted = f.read()
        except Exception as e:
            logger.error("Map '%s' update: read-back FAILED: %s", project_name, e)
            return False

        if len(persisted) != len(content):
            logger.error(
                "Map '%s' update: VERIFICATION FAILED — wrote %d chars but "
                "read back %d chars",
                project_name, len(content), len(persisted),
            )
            return False

        logger.info("Map '%s' updated and verified: %s (%d chars)", project_name, path, len(content))

        # Keep DB metadata (updated_at, content size) in sync with disk
        if self._store:
            self._store.update_map(project_name, content)

        return True

    async def delete_map(self, project_name: str) -> bool:
        """Delete a project map (file + DB metadata)."""
        path = self._map_file_path(project_name)
        file_deleted = False
        if path:
            try:
                os.remove(path)
                file_deleted = True
                logger.info("Map file deleted: %s", path)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error("Map file delete failed (%s): %s", path, e)

        db_deleted = self._store.delete_map(project_name)
        deleted = file_deleted or db_deleted
        if deleted:
            logger.info("Map deleted: '%s'", project_name)
            if self._active_project == project_name:
                self._active_project = None
        return deleted

    # ── Active Map Correction (tool-based, line-edit) ──────────

    async def apply_map_edit(
        self, project_name: str, old_text: str, new_text: str,
        replace_all: bool = False,
    ) -> str:
        """Exact string replacement in a project map.

        Returns confirmation string or error message.
        """
        content = await self.get_map(project_name)
        if content is None:
            return f"Error: no map found for project '{project_name}'"
        count = content.count(old_text)
        if count == 0:
            return f"Error: old_text not found in map '{project_name}'"
        if not replace_all and count > 1:
            return (
                f"Error: old_text matches {count} locations in map "
                f"'{project_name}' — provide a more specific string"
            )

        if replace_all:
            new_content = content.replace(old_text, new_text)
        else:
            new_content = content.replace(old_text, new_text, 1)
        saved = await self.update_map(project_name, new_content)
        if not saved:
            return f"Error: map edit computed but FAILED to persist for '{project_name}'"
        asyncio.create_task(self._update_map_summary_and_embedding(project_name, new_content))
        n_replaced = count if replace_all else 1
        return f"Map '{project_name}' updated successfully ({n_replaced} replacement{'s' if n_replaced > 1 else ''})."

    # ── Map Summary + Embedding ─────────────────────────────────

    @staticmethod
    def _extract_map_summary(map_content: str) -> str | None:
        """Extract the ## Summary section content from a project map."""
        lines = map_content.split("\n")
        in_summary = False
        summary_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("## Summary"):
                in_summary = True
                continue
            if in_summary and line.strip().startswith("## "):
                break
            if in_summary and line.strip():
                summary_lines.append(line.strip())
        return " ".join(summary_lines) if summary_lines else None

    async def _update_map_summary_and_embedding(
        self, project_name: str, map_content: str,
    ) -> None:
        """Regenerate summary and embedding for a project map.

        Called after every map content change (create, update, edit, curation).
        Best-effort — failures are logged but don't block the caller.
        """
        summary = self._extract_map_summary(map_content)
        if not summary:
            # Fallback: first non-heading paragraph
            lines = map_content.split("\n")
            para: list[str] = []
            for line in lines:
                if line.startswith("#"):
                    if para:
                        break
                    continue
                if line.strip() == "" and para:
                    break
                if line.strip():
                    para.append(line.strip())
            summary = " ".join(para)[:500] if para else None

        if not summary:
            return

        try:
            embedding = await self._embedder.embed_to_array(summary)
        except Exception:
            logger.warning("Failed to embed map summary for '%s'", project_name)
            embedding = None

        if self._store:
            self._store.update_map_summary(project_name, summary, embedding)
        self._map_embedding_cache.pop(project_name, None)
        logger.debug(
            "Map summary updated for '%s': %d chars, embedding=%s",
            project_name, len(summary), embedding is not None,
        )

    # ── Map Relevance Selection ──────────────────────────────────

    async def select_relevant_maps(
        self, context_text: str, k: int = 2, min_score: float = 0.2,
    ) -> list[tuple[str, float, str]]:
        """Select top-k project maps by relevance to conversation context.

        Returns list of (project_name, score, content) tuples, sorted by
        score descending. Falls back to empty list if embedder unavailable.
        """
        if not self._store or not context_text.strip():
            return []

        query_emb = await self._get_query_embedding(context_text)
        if query_emb is None:
            return []

        # Load map embeddings (cached in memory between updates)
        map_embs = self._load_map_embeddings()
        if not map_embs:
            return []

        # Compute cosine similarity
        scored: list[tuple[str, float]] = []
        for name, emb in map_embs.items():
            dot = float(np.dot(query_emb, emb))
            norm = float(np.linalg.norm(query_emb) * np.linalg.norm(emb))
            sim = dot / norm if norm > 0 else 0.0
            if sim >= min_score:
                scored.append((name, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:k]

        results: list[tuple[str, float, str]] = []
        for name, score in scored:
            content = await self.get_map(name)
            if content:
                results.append((name, score, content))

        if results:
            detail = ", ".join(f"{n} ({s:.2f})" for n, s, _ in results)
            logger.info("Maps selected: %s", detail)

        return results

    def _load_map_embeddings(self) -> dict[str, np.ndarray]:
        """Load map summary embeddings, using cache when available."""
        if self._map_embedding_cache:
            return self._map_embedding_cache

        if not self._store:
            return {}

        cache: dict[str, np.ndarray] = {}
        for name, _summary, blob in self._store.list_map_embeddings():
            if blob:
                emb = _deserialize_embedding(blob)
                if emb is not None:
                    cache[name] = emb
        self._map_embedding_cache = cache
        return cache

    # ── Initialization ─────────────────────────────────────────

    @staticmethod
    def _resolve_home(path: str) -> str:
        """Expand ~ to the real user home, not the CC agent's synthetic HOME."""
        from ..paths import resolve_path
        return resolve_path(path)

    async def set_project_context(
        self, project_dir: str, reset: bool = False
    ) -> str:
        """User-triggered project initialization.

        Accepts either a directory path or a project name. If the input is
        not a valid directory but matches an existing project name with a
        stored project_dir, uses the stored directory.

        Project name = os.path.basename(project_dir).
        If reset=True, deletes existing map and re-scans.
        If a map exists and reset=False, loads it and sets active.
        If no map exists, triggers exhaustive scan.

        Returns status message.
        """
        project_dir = self._resolve_home(project_dir)
        project_name = os.path.basename(os.path.normpath(project_dir))
        if not project_name:
            return "Error: could not derive project name from path"

        # If the resolved path isn't a directory, check if it's a known
        # project name with a stored project_dir
        if not os.path.isdir(project_dir):
            stored_dir = self._store.get_project_dir(project_name)
            if stored_dir and os.path.isdir(stored_dir):
                logger.info(
                    "Input '%s' is not a directory — using stored dir '%s' "
                    "for project '%s'",
                    project_dir, stored_dir, project_name,
                )
                project_dir = stored_dir
            elif not os.path.isdir(project_dir):
                # Check if the raw input (before resolve_home) matches a
                # project name directly
                raw_name = os.path.basename(project_dir)
                existing = self._store.get_map(raw_name)
                if existing:
                    stored = existing.get("project_dir")
                    if stored and os.path.isdir(stored):
                        logger.info(
                            "Matched project name '%s' — using stored dir '%s'",
                            raw_name, stored,
                        )
                        project_name = raw_name
                        project_dir = stored

        logger.info(
            "Project context set: '%s' (reset=%s)", project_name, reset
        )

        if reset:
            map_path = os.path.join(project_dir, "PROJECT_MAP.md")
            if os.path.exists(map_path):
                os.remove(map_path)
            self._store.delete_map(project_name)
            logger.info("Map deleted for reset: '%s'", project_name)

        # Check for existing map file on disk
        map_path = os.path.join(project_dir, "PROJECT_MAP.md")
        if os.path.exists(map_path) and not reset:
            # Load existing map, set active
            self._active_project = project_name
            self._active_project_dir = project_dir
            self._store.set_active_project(project_name)
            # Ensure metadata row exists in DB
            if not self._store.get_map(project_name):
                import sqlite3
                map_id = uuid.uuid4().hex[:12]
                try:
                    self._store.create_map(map_id, project_name, "", project_dir=project_dir)
                except sqlite3.IntegrityError:
                    pass
            return f"Project context loaded: {project_name}"

        # No map exists (or reset) — run scan
        content = await self._scan_project(project_dir, project_name)
        if content is None:
            return f"Error: scan failed for {project_dir}"

        ok = await self.create_map(project_name, content, project_dir=project_dir)
        if not ok:
            return f"Error: failed to write project map for {project_name}"
        self._active_project = project_name
        self._active_project_dir = project_dir
        self._store.set_active_project(project_name)
        return f"Project context initialized: {project_name}"

    async def _scan_project(
        self, project_dir: str, project_name: str
    ) -> str | None:
        """Run exhaustive project scan and synthesize a map.

        1. tree structure
        2. Prioritized file reads
        3. LLM synthesis
        4. Optional follow-up reads (up to 2 rounds)
        """
        import subprocess

        if not os.path.isdir(project_dir):
            logger.warning("Project dir not found: %s", project_dir)
            return None

        # Step 1: Directory tree
        try:
            result = subprocess.run(
                ["tree", "-L", "3", "--dirsfirst", "-I",
                 "node_modules|.venv|__pycache__|.git|.tox|dist|build|*.egg-info"],
                cwd=project_dir,
                capture_output=True, text=True, timeout=30,
            )
            tree_output = result.stdout[:8000]  # cap tree output
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback: basic ls
            tree_output = self._basic_tree(project_dir)

        # Step 2: Prioritized file reads
        file_contents = await self._read_priority_files(project_dir)

        # Step 3: LLM synthesis
        prompt = MAP_SYNTHESIS_PROMPT.format(
            project_name=project_name,
            tree_output=tree_output,
            file_contents=file_contents,
        )
        logger.info(
            "Map synthesis prompt: %d chars (%d tree, %d files)",
            len(prompt), len(tree_output), len(file_contents),
        )

        max_rounds = 2
        for round_num in range(max_rounds + 1):
            try:
                llm_response = await asyncio.wait_for(
                    self._llm_client.complete(prompt, max_tokens=16384),
                    timeout=300,
                )
            except asyncio.TimeoutError:
                logger.warning("Map synthesis LLM call timed out (round %d)", round_num)
                continue  # try next round (or exit loop)
            except Exception:
                logger.error("Map synthesis failed", exc_info=True)
                return None

            # Check for follow-up file requests
            match = re.search(
                r"<request_files>(.*?)</request_files>",
                llm_response, re.DOTALL,
            )
            if match and round_num < max_rounds:
                requested = [
                    f.strip() for f in match.group(1).split(",") if f.strip()
                ]
                # Also split on newlines (LLMs sometimes use one-per-line)
                expanded = []
                for r in requested:
                    expanded.extend(
                        p.strip() for p in r.split("\n") if p.strip()
                    )
                requested = expanded
                logger.info(
                    "Map synthesis round %d: LLM requested %d files: %s",
                    round_num + 1, len(requested), requested,
                )
                extra_content = self._read_requested_files(
                    project_dir, requested
                )
                if extra_content:
                    prompt += (
                        f"\n\n<additional_files>\n{extra_content}\n"
                        f"</additional_files>\n\nNow output the complete "
                        f"project map (no <request_files> block)."
                    )
                    continue
                else:
                    # Files not found — tell LLM to proceed without them
                    logger.warning(
                        "Map synthesis: requested files not found, "
                        "asking LLM to proceed without them"
                    )
                    prompt += (
                        "\n\nThe requested files could not be read. "
                        "Please produce the project map using only the "
                        "files already provided. Output ONLY the map "
                        "document (starting with # Project:)."
                    )
                    continue

            # Extract the map content (strip any request_files blocks)
            content = re.sub(
                r"<request_files>.*?</request_files>", "", llm_response,
                flags=re.DOTALL,
            ).strip()

            # Strip markdown code fences if present
            content = re.sub(
                r"^```(?:markdown|md)?\s*\n?", "", content,
            )
            content = re.sub(r"\n?```\s*$", "", content).strip()

            # Validate it starts with # Project:
            if not content.startswith("# Project:"):
                # Try to find it
                idx = content.find("# Project:")
                if idx >= 0:
                    content = content[idx:]
                else:
                    logger.warning(
                        "Map synthesis output missing '# Project:' header"
                    )

            logger.info(
                "Map '%s' synthesized: %d chars, %d words",
                project_name, len(content), len(content.split()),
            )
            return content

        return None

    def _basic_tree(self, project_dir: str, max_depth: int = 3) -> str:
        """Fallback directory listing when tree command is unavailable."""
        lines = []
        for root, dirs, files in os.walk(project_dir):
            depth = root.replace(project_dir, "").count(os.sep)
            if depth >= max_depth:
                dirs.clear()
                continue
            indent = "  " * depth
            lines.append(f"{indent}{os.path.basename(root)}/")
            for f in sorted(files)[:20]:
                lines.append(f"{indent}  {f}")
            # Skip common non-source dirs
            dirs[:] = [
                d for d in sorted(dirs)
                if d not in {
                    "node_modules", ".venv", "__pycache__", ".git",
                    ".tox", "dist", "build",
                }
            ]
            if len(lines) > 200:
                lines.append("... (truncated)")
                break
        return "\n".join(lines)

    async def _read_priority_files(
        self, project_dir: str, budget_chars: int = 60_000
    ) -> str:
        """Read prioritized files from a project directory."""
        import glob as glob_mod

        parts = []
        total_chars = 0

        # Collect candidate files in priority order
        candidates: list[str] = []
        for category in ["docs", "config", "entry_points", "init_files"]:
            for pattern in _PRIORITY_GLOBS[category]:
                matches = glob_mod.glob(
                    os.path.join(project_dir, pattern), recursive=False
                )
                matches += glob_mod.glob(
                    os.path.join(project_dir, "**", pattern), recursive=True
                )
                candidates.extend(matches)

        # Also add largest .py files
        py_files = glob_mod.glob(
            os.path.join(project_dir, "**", "*.py"), recursive=True
        )
        # Skip common non-source dirs
        skip_dirs = [
            "node_modules", ".venv", "__pycache__",
            ".git", ".tox", "dist", "build",
            "checkpoints", "results", "data", "logs",
        ]
        py_files = [
            f for f in py_files
            if not any(
                f"/{skip}/" in f or f.endswith(f"/{skip}")
                for skip in skip_dirs
            )
        ]
        py_files.sort(key=lambda f: os.path.getsize(f), reverse=True)
        candidates.extend(py_files[:15])

        # Deduplicate while preserving order; filter skip_dirs
        seen = set()
        unique_candidates = []
        for f in candidates:
            real = os.path.realpath(f)
            if real in seen or not os.path.isfile(real):
                continue
            if any(f"/{skip}/" in real for skip in skip_dirs):
                continue
            seen.add(real)
            unique_candidates.append(real)

        for filepath in unique_candidates:
            if total_chars >= budget_chars:
                break
            try:
                with open(filepath, "r", errors="replace") as fh:
                    content = fh.read(20_000)  # cap per file
                rel_path = os.path.relpath(filepath, project_dir)
                chunk = f"\n--- {rel_path} ---\n{content}\n"
                parts.append(chunk)
                total_chars += len(chunk)
            except (OSError, UnicodeDecodeError):
                continue

        logger.debug(
            "Read %d priority files (%d chars) from %s",
            len(parts), total_chars, project_dir,
        )
        return "".join(parts)

    def _read_requested_files(
        self, project_dir: str, paths: list[str], budget_chars: int = 50_000
    ) -> str:
        """Read files requested by the LLM during synthesis."""
        real_project = os.path.realpath(project_dir)
        parts = []
        total = 0
        for raw_path in paths:
            if total >= budget_chars:
                break
            # Clean up LLM path formatting quirks
            rel_path = raw_path.strip().strip("`'\"")
            rel_path = rel_path.lstrip("./")
            # If the LLM gave an absolute path, try to make it relative
            if rel_path.startswith("/"):
                if rel_path.startswith(real_project + "/"):
                    rel_path = rel_path[len(real_project) + 1:]
                else:
                    logger.debug(
                        "_read_requested_files: skipping absolute path outside project: %s",
                        raw_path,
                    )
                    continue
            filepath = os.path.normpath(os.path.join(project_dir, rel_path))
            # Safety: stay within project dir
            if not filepath.startswith(real_project):
                logger.debug(
                    "_read_requested_files: path escapes project dir: %s → %s",
                    raw_path, filepath,
                )
                continue
            if not os.path.isfile(filepath):
                logger.warning(
                    "_read_requested_files: file not found: %s (resolved: %s)",
                    raw_path, filepath,
                )
                continue
            try:
                with open(filepath, "r", errors="replace") as fh:
                    content = fh.read(20_000)
                chunk = f"\n--- {rel_path} ---\n{content}\n"
                parts.append(chunk)
                total += len(chunk)
                logger.debug("_read_requested_files: read %s (%d chars)", rel_path, len(content))
            except (OSError, UnicodeDecodeError) as e:
                logger.warning("_read_requested_files: error reading %s: %s", filepath, e)
                continue
        if not parts and paths:
            logger.warning(
                "_read_requested_files: none of %d requested files could be read: %s",
                len(paths), paths,
            )
        return "".join(parts)

    # ── Phase 2: Window Drop Pipeline ────────────────────────

    async def on_window_drop(self, old_half_turns: list) -> None:
        """Process dropped turns from rolling window.

        Pipeline:
        1. Segment old_half by topic (consecutive same-label groups)
        2. For each segment: significance gate → LLM reflection → log entry
        3. Map curation on raw dropped turns (active project only)
        4. Conversation summary update (LLM compresses dropped turns + existing summary)
        5. New project detection (if reflection names unknown project)

        Called by router_v2._check_and_trigger_summarization() for v2.
        """
        if not old_half_turns:
            logger.debug("on_window_drop called with empty turns, skipping")
            return

        # When v3 is enabled, memory formation, map curation, and new-project
        # detection are driven by the unified `form_un_formed` operation
        # (time-based / token-pressure / shutdown / startup triggers; rev 5
        # dropped window-drop as a formation trigger). Only the rolling-window
        # conversation summary continues to fire from this hook — it is
        # orthogonal to memory formation.
        if self._formation_v3_enabled:
            logger.debug(
                "v3 enabled: short-circuiting window-drop formation/curation; "
                "running rolling-window conversation summary only "
                "(%d dropped turns)",
                len(old_half_turns),
            )
            await self._update_conversation_summary(
                old_half_turns, len(old_half_turns),
            )
            return

        logger.info(
            "Window drop pipeline starting: %d turns to process",
            len(old_half_turns),
        )

        # 1. Segment by topic
        segments = self._segment_by_topic(old_half_turns)
        logger.info(
            "Topic segmentation: %d segments from %d turns",
            len(segments), len(old_half_turns),
        )

        # 2. Known project names for reflection prompt
        known_projects = self._known_project_names()

        # 3. For each segment: reflect → store.
        # Per user directive (2026-04-26): memory formation must trigger on
        # every window drop, regardless of map status. The previous
        # _significance_gate filter was rejecting ~98% of segments (1,478
        # below / 24 passed across all agents). Topic segmentation already
        # groups consecutive same-label turns, so a window drop typically
        # produces only a handful of segments — let each become a memory.
        # _significance_gate is retained as a rule library for callers that
        # still want it (and for unit tests), but is no longer applied here.
        new_entries = []
        for topic_label, turns in segments:
            if not turns:
                continue
            logger.info(
                "Segment '%s' (%d turns) reflecting (no significance gate)",
                topic_label, len(turns),
            )
            entry = await self._reflect_on_segment(
                turns, topic_label, known_projects,
            )
            if entry:
                new_entries.append(entry)
                logger.info(
                    "Log entry created: id=%s project='%s' topic='%s' outcome='%s'",
                    entry.id, entry.project, entry.topic_label, entry.outcome,
                )

        # 4. Map curation (if active project exists)
        if self._active_project:
            raw_text = self._format_turns_as_text(old_half_turns)
            await self._curate_active_map(raw_text, len(old_half_turns))

        # 5. Conversation summary update
        await self._update_conversation_summary(
            old_half_turns, len(old_half_turns),
        )

        # 6. New project detection
        for entry in new_entries:
            if entry.project and entry.project != (self._active_project or ""):
                existing = self._store.get_map(entry.project)
                if not existing:
                    logger.info(
                        "New project detected: '%s' — bootstrapping map from %d turns",
                        entry.project, len(old_half_turns),
                    )
                    await self._create_new_project_map(
                        entry.project, entry.summary,
                    )

        logger.info(
            "Window drop pipeline complete: %d segments, %d log entries created",
            len(segments), len(new_entries),
        )

    @staticmethod
    def _segment_by_topic(turns: list) -> list[tuple[str, list]]:
        """Group turns by consecutive topic_label.

        Returns list of (topic_label, [turns]) tuples.
        Turns without a label are grouped with their neighbors
        or form a "misc" segment.
        """
        if not turns:
            return []

        segments: list[tuple[str, list]] = []
        current_label = ""
        current_group: list = []

        for turn in turns:
            label = getattr(turn, "meta", {}).get("topic_label", "") or ""
            if not label:
                # Unlabeled turn — attach to current group
                current_group.append(turn)
            elif label == current_label:
                current_group.append(turn)
            else:
                # New label — flush current group
                if current_group:
                    segments.append((current_label or "misc", current_group))
                current_label = label
                current_group = [turn]

        # Flush final group
        if current_group:
            segments.append((current_label or "misc", current_group))

        return segments

    def _significance_gate(self, turns: list) -> bool:
        """Rule-based filter: is this segment worth a log entry?

        Criteria (same as should_reflect):
        - Tool-heavy: tool_calls >= reflection_min_tools (default 3)
        - Extended discussion: turns >= 4 AND chars >= 1500
        - Brainstorm: tools <= 2 AND agent_chars >= 1500 AND turns >= 2
        - Errors: any errors in the segment
        """
        tool_calls = 0
        user_visible_turns = 0
        total_chars = 0
        agent_chars = 0
        has_errors = False

        for turn in turns:
            content = getattr(turn, "content", "")
            content_str = content if isinstance(content, str) else str(content)
            role = getattr(turn, "role", "")
            meta = getattr(turn, "meta", {}) or {}
            from_node = getattr(turn, "from_node", "")

            # Count tool calls
            if meta.get("tool_calls"):
                calls = meta["tool_calls"]
                if isinstance(calls, list):
                    tool_calls += len(calls)
                else:
                    tool_calls += 1
            if meta.get("cc_tool_events"):
                cc_count = meta.get("cc_tool_calls", 0)
                tool_calls += cc_count if cc_count else 1

            # Count tool-role turns as tool calls too
            if role == "tool":
                tool_calls += 1
                # Check for errors in tool results
                if "error" in content_str.lower()[:500]:
                    has_errors = True
                continue

            # User-visible counting
            if role in ("user", "assistant") or from_node.startswith("user:"):
                user_visible_turns += 1
                total_chars += len(content_str)
                if role == "assistant" or from_node.startswith("agent:"):
                    agent_chars += len(content_str)

        # Apply gates
        if tool_calls >= self._reflection_min_tools:
            logger.debug("Significance gate: tool-heavy (%d tools)", tool_calls)
            return True

        if (user_visible_turns >= self._reflection_min_discussion_turns
                and total_chars >= self._reflection_min_discussion_chars):
            logger.debug(
                "Significance gate: extended discussion (%d turns, %d chars)",
                user_visible_turns, total_chars,
            )
            return True

        if (tool_calls <= self._reflection_max_brainstorm_tools
                and agent_chars >= self._reflection_min_brainstorm_response_chars
                and user_visible_turns >= 2):
            logger.debug(
                "Significance gate: brainstorm (%d agent chars, %d turns)",
                agent_chars, user_visible_turns,
            )
            return True

        if has_errors:
            logger.debug("Significance gate: errors detected")
            return True

        return False

    async def _reflect_on_segment(
        self, turns: list, topic_label: str, known_projects: str,
    ) -> MemoryEntry | None:
        """Run LLM reflection on a segment and create a log entry."""
        # Build trigger (first user message in segment)
        trigger = f"[TOPIC: {topic_label}]"
        for turn in turns:
            from_node = getattr(turn, "from_node", "")
            role = getattr(turn, "role", "")
            if role == "user" or from_node.startswith("user:"):
                content = getattr(turn, "content", "")
                trigger = f"[TOPIC: {topic_label}] {content[:500]}"
                break

        # Build trace from turns
        trace = self._format_turns_as_trace(turns)

        # Call LLM
        prompt = WINDOW_DROP_REFLECTION_PROMPT.format(
            trigger=trigger,
            trace=trace,
            known_project_names=known_projects,
        )

        try:
            llm_response = await asyncio.wait_for(
                self._llm_client.complete(prompt, max_tokens=16384),
                timeout=180,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Window drop reflection timed out for segment '%s'", topic_label,
            )
            return None
        except Exception:
            logger.error(
                "Window drop reflection failed for segment '%s'",
                topic_label, exc_info=True,
            )
            return None

        # Parse response
        reflection = _extract_tag(llm_response, "reflection")
        summary = _extract_tag(llm_response, "summary")
        tags_str = _extract_tag(llm_response, "tags")
        outcome_label = _extract_tag(llm_response, "outcome_label")
        retrieval_key = _extract_tag(llm_response, "retrieval_key")
        project = _extract_tag(llm_response, "project")

        if not reflection or not summary:
            logger.warning(
                "Window drop reflection produced empty output for segment '%s'",
                topic_label,
            )
            return None

        if not retrieval_key:
            retrieval_key = summary

        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        outcome = outcome_label.strip() if outcome_label else "success"
        if outcome not in ("success", "partial", "failure"):
            outcome = "success"

        # Classify project (sanity check)
        if not project:
            project = self._active_project or ""
        project = project.strip()

        # Embed reflection + retrieval_key
        try:
            emb_results = await asyncio.wait_for(
                self._embedder.embed_batch_to_arrays([reflection, retrieval_key]),
                timeout=30,
            )
        except Exception:
            logger.error("Embedding failed for segment '%s'", topic_label, exc_info=True)
            emb_results = [None, None]

        entry = MemoryEntry(
            id=MemoryEntry.new_id(),
            created_at=datetime.now(timezone.utc),
            summary=summary,
            reflection=reflection,
            trace=trace[:self._trace_max_tokens * 4],  # clip trace
            trigger=trigger,
            retrieval_key=retrieval_key,
            topic_label=topic_label,
            tags=tags,
            outcome=outcome,
            reflection_embedding=emb_results[0],
            retrieval_key_embedding=emb_results[1],
            weight=0.0,
            project=project,
        )

        self._pool.append(entry)
        self._store.insert(entry)
        self._incremental_active_update(entry)
        self._prune_pool()

        return entry

    def _format_turns_as_trace(self, turns: list) -> str:
        """Format turns as a tool-call trace for reflection prompts."""
        lines = []
        char_budget = self._trace_max_tokens * 4

        for turn in turns:
            meta = getattr(turn, "meta", {}) or {}
            content = getattr(turn, "content", "")
            role = getattr(turn, "role", "")

            if meta.get("tool_calls"):
                calls = meta["tool_calls"]
                if isinstance(calls, list):
                    for call in calls:
                        if isinstance(call, dict):
                            name = call.get("name", call.get("tool_name", "?"))
                            args = str(call.get("arguments", call.get("args", "")))
                            lines.append(f"[TOOL] {name}({args[:200]})")
                elif isinstance(calls, str):
                    lines.append(f"[TOOL] {calls[:300]}")
            elif role == "tool" or meta.get("tool_results"):
                result_str = str(content)[:500]
                lines.append(f"[RESULT] {result_str}")
            elif role in ("user", "assistant"):
                from_node = getattr(turn, "from_node", "")
                prefix = "USER" if (role == "user" or from_node.startswith("user:")) else "AGENT"
                lines.append(f"[{prefix}] {str(content)[:1000]}")

        trace = "\n".join(lines)
        if len(trace) > char_budget:
            trace = trace[:char_budget] + "\n... (trace truncated)"
        return trace

    @staticmethod
    def _format_turns_as_text(turns: list) -> str:
        """Format turns as readable conversation text for map curation.

        Separates user messages (high priority) from agent/tool messages
        to ensure user-provided definitions and decisions aren't drowned
        out by verbose agent output and tool call traces.
        """
        user_lines = []
        agent_lines = []
        seen_content = set()

        for turn in turns:
            role = getattr(turn, "role", "unknown")
            content = str(getattr(turn, "content", ""))
            from_node = getattr(turn, "from_node", "")
            to_node = getattr(turn, "to_node", "")
            ts = getattr(turn, "timestamp", "")

            # Skip internal tool activity logs entirely
            if to_node == "internal" or "[CC Tool Activity]" in content:
                continue
            if content.startswith("[Tool:") or content.startswith("[cc:"):
                continue
            if from_node == "system":
                continue

            # Deduplicate
            dedup_key = (from_node, content[:200])
            if dedup_key in seen_content:
                continue
            seen_content.add(dedup_key)

            # Determine display role
            if from_node.startswith("user:"):
                display = "User"
            elif from_node.startswith("agent:"):
                display = f"Agent ({from_node.split(':')[-1]})" if ":" in from_node else "Agent"
            elif role == "tool":
                display = "Tool Result"
            else:
                display = role.capitalize()

            ts_str = ""
            if ts:
                if hasattr(ts, "strftime"):
                    ts_str = ts.strftime("%H:%M")
                elif isinstance(ts, str) and len(ts) >= 16:
                    ts_str = ts[11:16]

            content_str = content[:3000]
            if ts_str:
                line = f"[{ts_str}] {display}: {content_str}"
            else:
                line = f"{display}: {content_str}"

            if from_node.startswith("user:"):
                user_lines.append(line)
            else:
                agent_lines.append(line)

        # Cap agent text to avoid drowning out user messages
        agent_text = "\n\n".join(agent_lines)
        if len(agent_text) > 15000:
            agent_text = agent_text[:15000] + "\n\n[... agent output truncated ...]"

        parts = []
        if user_lines:
            parts.append(
                "=== USER MESSAGES (high priority — defines vocabulary, "
                "decisions, requirements) ===\n\n"
                + "\n\n".join(user_lines)
            )
        if agent_text.strip():
            parts.append(
                "=== AGENT RESPONSES (supporting context) ===\n\n"
                + agent_text
            )
        return "\n\n".join(parts)

    async def _update_conversation_summary(
        self, dropped_turns: list, turn_count: int,
    ) -> None:
        """Generate or update the conversation summary from dropped turns.

        Uses the existing summary (if any) + formatted dropped turns to produce
        an updated narrative. The summary captures the conversational thread —
        who asked what, what happened, what's pending — complementing the
        reflections (lessons learned) and maps (project state).

        Called from on_window_drop alongside map curation.
        """
        raw_text = self._format_turns_as_text(dropped_turns)

        # Build prompt
        if self._conversation_summary:
            existing_section = (
                f"[Existing summary]\n{self._conversation_summary}"
            )
        else:
            existing_section = "[No existing summary — this is the first batch]"

        prompt = CONVERSATION_SUMMARY_PROMPT.format(
            existing_summary_section=existing_section,
            new_turns=raw_text,
        )

        try:
            t0 = time.time()
            summary_text = await self._llm_client.complete(
                prompt, max_tokens=4096,
            )
            elapsed = time.time() - t0

            # Enforce token cap
            from ..llm import estimate_tokens
            token_count = estimate_tokens(summary_text)
            if token_count > SUMMARY_MAX_TOKENS:
                # Truncate from the beginning (oldest content)
                char_budget = SUMMARY_MAX_TOKENS * 4
                if len(summary_text) > char_budget:
                    summary_text = "...\n" + summary_text[-char_budget:]
                    token_count = estimate_tokens(summary_text)
                logger.warning(
                    "Summary exceeded cap (%d > %d tokens), truncated",
                    token_count, SUMMARY_MAX_TOKENS,
                )

            total_msgs = self._summary_messages_summarized + turn_count
            self._conversation_summary = summary_text
            self._summary_messages_summarized = total_msgs

            # Persist to DB
            self._store.save_summary(summary_text, total_msgs, token_count)

            logger.info(
                "Conversation summary updated: %d chars (~%d tokens), "
                "%d total messages covered, %.1fs",
                len(summary_text), token_count, total_msgs, elapsed,
            )
        except Exception:
            logger.error(
                "Conversation summary generation failed", exc_info=True,
            )

    async def _curate_active_map(
        self, raw_turns_text: str, input_turn_count: int = 0,
    ) -> None:
        """Run map curation on the active project using raw dropped turns.

        Supports tool calls (file_read, bash_exec) for resolving pre-existing
        contradictions, capped by _curation_audit_max_tool_calls.
        """
        if not self._active_project:
            return

        current_map = await self.get_map(self._active_project)
        if not current_map:
            logger.warning(
                "Active project '%s' has no map, skipping curation",
                self._active_project,
            )
            return

        old_tokens = len(current_map.split())

        # Logging contract: curation started
        raw_tokens = len(raw_turns_text.split())
        logger.info(
            "Curating map '%s': %d input turns, %d tokens of raw context",
            self._active_project, input_turn_count, raw_tokens,
        )

        prompt = MAP_CURATION_PROMPT.format(
            current_map=current_map,
            raw_conversation=raw_turns_text,
        )

        tool_calls_used = 0

        try:
            # Tool call loop: LLM may request file_read/bash_exec
            for _round in range(self._curation_audit_max_tool_calls + 1):
                llm_response = await asyncio.wait_for(
                    self._llm_client.complete(prompt, max_tokens=16384),
                    timeout=180,
                )

                # Check for tool calls in response
                calls = _parse_tool_calls(llm_response)
                if not calls or tool_calls_used >= self._curation_audit_max_tool_calls:
                    break

                # Execute tool calls and build continuation prompt
                tool_results = []
                for tool_name, arg in calls:
                    if tool_calls_used >= self._curation_audit_max_tool_calls:
                        break
                    tool_calls_used += 1
                    logger.debug(
                        "Curation tool call: %s(%s)",
                        tool_name, arg[:200],
                    )
                    result = await _execute_tool_call(
                        tool_name, arg, self._active_project_dir,
                    )
                    tool_results.append(
                        f"<tool_result tool=\"{tool_name}\" arg=\"{arg}\">\n"
                        f"{result}\n</tool_result>"
                    )

                # Continue with tool results appended
                prompt = (
                    prompt + "\n\n" + llm_response + "\n\n"
                    + "\n".join(tool_results)
                    + "\n\nContinue with the map output based on the tool results above."
                )

        except asyncio.TimeoutError:
            logger.warning(
                "Map '%s' curation failed: timed out", self._active_project,
            )
            return
        except Exception as e:
            logger.warning(
                "Map '%s' curation failed: %s", self._active_project, e,
            )
            return

        # Check for no-op response
        response_stripped = llm_response.strip().lower()
        if "no updates needed" in response_stripped[:50]:
            logger.info(
                "Map '%s' curation: no updates needed — dropped turns irrelevant",
                self._active_project,
            )
            return

        # Apply section-level updates
        updated = _apply_section_updates(current_map, llm_response)
        if updated:
            content = updated
        else:
            # Fallback: treat response as complete map (legacy format)
            content = re.sub(r"^```(?:markdown|md)?\s*\n?", "", llm_response.strip())
            content = re.sub(r"\n?```\s*$", "", content).strip()
            if not content.startswith("# Project:"):
                idx = content.find("# Project:")
                if idx >= 0:
                    content = content[idx:]
                else:
                    logger.warning(
                        "Map '%s' curation failed: no section tags and no '# Project:' header",
                        self._active_project,
                    )
                    return

        saved = await self.update_map(self._active_project, content)
        if saved:
            asyncio.create_task(
                self._update_map_summary_and_embedding(self._active_project, content)
            )

        # Logging contract: curation completed
        new_tokens = len(content.split())
        logger.info(
            "Map '%s' curated: %d → %d tokens (%+d), tool_calls=%d",
            self._active_project, old_tokens, new_tokens,
            new_tokens - old_tokens, tool_calls_used,
        )

    # ── Entry-Driven Map Integration ─────────────────────────────

    @staticmethod
    def _format_entries_for_integration(entries: list[MemoryEntry]) -> str:
        """Format memory entries as text for the integration prompt."""
        parts = []
        for entry in entries:
            score_tag = next(
                (t for t in entry.tags if t.startswith("score:")), None,
            )
            score_str = score_tag.split(":", 1)[1] if score_tag else "?"
            non_score_tags = [t for t in entry.tags if not t.startswith("score:")]
            parts.append(
                f"### Entry: {entry.retrieval_key or entry.topic_label}\n"
                f"- Date: {entry.created_at.strftime('%Y-%m-%d')}\n"
                f"- Topic: {entry.topic_label}\n"
                f"- Score: {score_str}/10\n"
                f"- Outcome: {entry.outcome}\n"
                f"- Tags: {', '.join(non_score_tags) if non_score_tags else 'none'}\n"
                f"- Summary: {entry.summary}"
            )
        return "\n\n".join(parts)

    async def _integrate_entries_into_maps(
        self, entries: list[MemoryEntry],
    ) -> None:
        """Integrate newly persisted memory entries into their project maps.

        Groups entries by project and runs one LLM call per project to patch
        the map with new information. Fire-and-forget — all failures are
        non-fatal (entries are already persisted).
        """
        by_project: dict[str, list[MemoryEntry]] = {}
        for entry in entries:
            if entry.project:
                by_project.setdefault(entry.project, []).append(entry)

        if not by_project:
            return

        for project, proj_entries in by_project.items():
            try:
                await self._integrate_project_entries(project, proj_entries)
            except Exception as e:
                logger.warning(
                    "Map integration failed for '%s': %s", project, e,
                )

    async def _integrate_project_entries(
        self, project_name: str, entries: list[MemoryEntry],
    ) -> None:
        """Integrate a batch of entries into a single project's map."""
        current_map = await self.get_map(project_name)
        if not current_map:
            logger.info(
                "Map integration skipped for '%s': no map exists",
                project_name,
            )
            return

        old_tokens = len(current_map.split())
        entries_text = self._format_entries_for_integration(entries)

        prompt = ENTRY_INTEGRATION_PROMPT.format(
            current_map=current_map,
            entries_text=entries_text,
        )

        llm = self._formation_llm_client or self._llm_client
        try:
            llm_response = await asyncio.wait_for(
                llm.complete(prompt, max_tokens=32768),
                timeout=120,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Map integration timed out for '%s' (%d entries)",
                project_name, len(entries),
            )
            return

        response_stripped = llm_response.strip().lower()
        if "no updates needed" in response_stripped[:50]:
            logger.info(
                "Map integration for '%s': no updates needed (%d entries)",
                project_name, len(entries),
            )
            return

        updated = _apply_section_updates(current_map, llm_response)
        if not updated:
            logger.warning(
                "Map integration for '%s': no valid section tags in LLM response",
                project_name,
            )
            return

        saved = await self.update_map(project_name, updated)
        if saved:
            asyncio.create_task(
                self._update_map_summary_and_embedding(project_name, updated)
            )
            new_tokens = len(updated.split())
            logger.info(
                "Map integration for '%s': %d → %d tokens (%+d) from %d entries",
                project_name, old_tokens, new_tokens,
                new_tokens - old_tokens, len(entries),
            )
        else:
            logger.error(
                "Map integration for '%s': update_map failed", project_name,
            )

    async def _audit_map_consistency(
        self, project_name: str, map_content: str,
    ) -> list[str]:
        """Standalone consistency audit — for use outside the curation pipeline.

        The main curation pipeline handles consistency as part of its single
        combined prompt (Step 2 in the curation instruction). This method exists
        for on-demand auditing (e.g., after multiple rapid map_edit calls) and
        for testing.

        Returns a list of inconsistency descriptions, or empty list if consistent.
        """
        prompt = MAP_AUDIT_PROMPT.format(map_content=map_content)

        try:
            llm_response = await asyncio.wait_for(
                self._llm_client.complete(prompt, max_tokens=4096),
                timeout=60,
            )
        except Exception as e:
            logger.warning(
                "Map '%s' audit failed: %s", project_name, e,
            )
            return []

        response_stripped = llm_response.strip().lower()
        if "no issues found" in response_stripped[:30]:
            return []

        # Parse bullet-point inconsistencies
        issues = []
        for line in llm_response.strip().splitlines():
            line = line.strip()
            if line.startswith("- "):
                issues.append(line[2:].strip())
            elif line and not line.startswith("#"):
                issues.append(line)
        return issues

    async def review_active_map(
        self, project_dir: str | None = None, recent_turns_text: str = "",
    ) -> dict:
        """Full map review: reconcile active map against the project filesystem.

        Performs a fresh project scan (tree + priority file reads) and asks
        the LLM to compare the current map against the actual project state,
        fixing discrepancies and surfacing ambiguities the user must resolve.

        project_dir defaults to the last directory used with set_project_context.

        Returns dict with:
          - "updated": bool — whether the map was changed
          - "summary": str — human-readable summary of what changed
        """
        if not self._active_project:
            return {
                "updated": False,
                "summary": "No active project. Use set_project_context first.",
            }

        current_map = await self.get_map(self._active_project)
        if not current_map:
            return {
                "updated": False,
                "summary": f"No map found for active project '{self._active_project}'.",
            }

        # Resolve project_dir — prefer explicit arg, fall back to cached dir,
        # then stored dir in DB
        resolved_dir = self._resolve_home(project_dir) if project_dir else self._active_project_dir
        if not resolved_dir:
            resolved_dir = self._store.get_project_dir(self._active_project)
        if not resolved_dir:
            return {
                "updated": False,
                "summary": (
                    "No project directory known. Provide project_dir or call "
                    "set_project_context first."
                ),
            }
        project_dir = resolved_dir
        if not os.path.isdir(project_dir):
            return {
                "updated": False,
                "summary": f"Project directory not found: {project_dir}",
            }

        old_tokens = len(current_map.split())
        logger.info(
            "Map review starting for '%s' from %s (%d tokens)",
            self._active_project, project_dir, old_tokens,
        )

        # Step 1: Build a short orientation hint (top-level tree, ~50 lines)
        orientation = ""
        import subprocess
        _skip_dirs = {
            "node_modules", ".venv", "__pycache__", ".git", ".tox", "dist",
            "build", "checkpoints", "data", "output", "results", "logs", "wandb",
        }
        try:
            result = subprocess.run(
                ["tree", "-L", "2", "--dirsfirst", "-I",
                 "|".join(_skip_dirs)],
                cwd=project_dir,
                capture_output=True, text=True, timeout=15,
            )
            tree_lines = result.stdout.strip().splitlines()[:60]
            if tree_lines:
                orientation = (
                    "<orientation_hint>\n"
                    "Top-level structure (depth 2, data dirs excluded). "
                    "Use tools to explore deeper.\n"
                    + "\n".join(tree_lines) + "\n"
                    "</orientation_hint>"
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # Fall through to Python fallback below

        # Python fallback when tree is not installed
        if not orientation:
            lines = []
            for root, dirs, files in os.walk(project_dir):
                depth = root.replace(project_dir, "").count(os.sep)
                if depth >= 2:
                    dirs.clear()
                    continue
                dirs[:] = sorted(d for d in dirs if d not in _skip_dirs
                                 and not d.endswith(".egg-info"))
                indent = "  " * depth
                base = os.path.basename(root) or os.path.basename(project_dir)
                if depth > 0:
                    lines.append(f"{indent}{base}/")
                else:
                    lines.append(f"{base}/")
                for f in sorted(files):
                    lines.append(f"{indent}  {f}")
                if len(lines) >= 60:
                    break
            if lines:
                orientation = (
                    "<orientation_hint>\n"
                    "Top-level structure (depth 2, data dirs excluded). "
                    "Use tools to explore deeper.\n"
                    + "\n".join(lines[:60]) + "\n"
                    "</orientation_hint>"
                )

        # Step 2: Build recent context block (optional)
        recent_context = ""
        if recent_turns_text:
            recent_context = (
                "<recent_conversation>\n"
                f"{recent_turns_text[:30000]}\n"
                "</recent_conversation>"
            )

        # Step 3: Build prompt — LLM explores interactively via tools
        prompt = MAP_REVIEW_PROMPT.format(
            current_map=current_map,
            project_dir=project_dir,
            orientation=orientation,
            recent_context=recent_context,
        )
        logger.info(
            "Map review prompt: %d chars (interactive mode, budget=%d tool calls)",
            len(prompt), self._review_max_tool_calls,
        )

        # Step 4: LLM tool loop — interactive exploration + map update
        tool_calls_used = 0
        llm_response = ""
        max_calls = self._review_max_tool_calls
        _rejected_premature = False
        try:
            for _round in range(max_calls + 1):
                llm_response = await asyncio.wait_for(
                    self._llm_client.complete(prompt, max_tokens=16384),
                    timeout=300,
                )

                calls = _parse_tool_calls(llm_response)
                logger.info(
                    "Review round %d: %d chars response, %d tool calls parsed, "
                    "total_used=%d. Has <updated_map>: %s",
                    _round, len(llm_response), len(calls),
                    tool_calls_used + len(calls),
                    "<updated_map>" in llm_response,
                )

                # Guard: reject early map if the LLM hasn't read any files yet
                # (only applies when orientation hint shows there are files to read;
                # fires at most once via _rejected_premature flag)
                if (not calls and tool_calls_used == 0
                        and "<updated_map>" in llm_response
                        and orientation
                        and not _rejected_premature):
                    _rejected_premature = True
                    logger.info(
                        "Review round %d: rejecting premature map "
                        "(0 tool calls used). Asking LLM to explore first.",
                        _round,
                    )
                    prompt = (
                        prompt + "\n\n" + llm_response
                        + "\n\nYou produced a map without reading any source files. "
                        "The orientation hint only shows the directory tree — "
                        "it does NOT show file contents. You MUST read the actual "
                        "files (models, configs, scripts, requirements.txt) to "
                        "verify specific values before producing the map.\n\n"
                        "Use <file_read>path</file_read> to read the key files now."
                    )
                    llm_response = ""
                    continue

                if not calls or tool_calls_used >= max_calls:
                    break

                tool_results = []
                for tool_name, arg in calls:
                    if tool_calls_used >= max_calls:
                        break
                    tool_calls_used += 1
                    logger.debug("Review tool call [%d/%d]: %s(%s)",
                                 tool_calls_used, max_calls, tool_name, arg[:200])
                    result_text = await _execute_tool_call(tool_name, arg, project_dir)
                    tool_results.append(
                        f"<tool_result tool=\"{tool_name}\" arg=\"{arg}\">\n"
                        f"{result_text}\n</tool_result>"
                    )

                budget_note = ""
                if tool_calls_used >= max_calls - 5:
                    budget_note = (
                        f"\n\nNote: {max_calls - tool_calls_used} tool calls remaining. "
                        "Wrap up exploration and produce the <updated_map>."
                    )

                prompt = (
                    prompt
                    + "\n\n[ASSISTANT]\n" + llm_response
                    + "\n\n[TOOL RESULTS]\n" + "\n".join(tool_results)
                    + "\n\n[INSTRUCTIONS]\nThe tool calls above were executed "
                    "successfully and the results are shown. Continue using "
                    "<file_read>path</file_read> or <bash_exec>cmd</bash_exec> "
                    "to explore more files, or produce the <updated_map> when ready."
                    + budget_note
                )

        except asyncio.TimeoutError:
            logger.warning("Map review timed out for '%s'", self._active_project)
            return {
                "updated": False,
                "summary": "Map review timed out. Try again later.",
            }
        except Exception as e:
            logger.warning("Map review failed for '%s': %s", self._active_project, e)
            return {
                "updated": False,
                "summary": f"Map review failed: {e}",
            }

        # Step 5: Parse response — extract <updated_map>
        updated_map = _extract_tag(llm_response, "updated_map")

        # Step 6: Validate and save the updated map
        if not updated_map:
            logger.warning(
                "Map review for '%s': no <updated_map> block in response",
                self._active_project,
            )
            return {
                "updated": False,
                "summary": "Review completed but could not extract updated map from LLM response.",
            }

        # Strip markdown fences
        updated_map = re.sub(r"^```(?:markdown|md)?\s*\n?", "", updated_map.strip())
        updated_map = re.sub(r"\n?```\s*$", "", updated_map).strip()

        # Validate header
        if not updated_map.startswith("# Project:"):
            idx = updated_map.find("# Project:")
            if idx >= 0:
                updated_map = updated_map[idx:]
            else:
                logger.warning(
                    "Map review for '%s': output missing '# Project:' header",
                    self._active_project,
                )
                return {
                    "updated": False,
                    "summary": "Review produced invalid map (missing '# Project:' header).",
                }

        # Check if map actually changed
        if updated_map.strip() == current_map.strip():
            logger.info("Map review for '%s': no changes needed", self._active_project)
            return {
                "updated": False,
                "summary": "Map is up to date — no changes needed.",
            }

        # Save with verification
        saved = await self.update_map(self._active_project, updated_map)
        new_tokens = len(updated_map.split())

        if not saved:
            logger.error(
                "Map review for '%s': LLM produced valid update (%d tokens) "
                "but PERSISTENCE FAILED — map was NOT saved",
                self._active_project, new_tokens,
            )
            return {
                "updated": False,
                "summary": (
                    f"PERSISTENCE FAILURE: Review produced a valid map update "
                    f"({old_tokens} → {new_tokens} tokens) but the write to "
                    f"the database failed. The map was NOT updated. "
                    f"Check logs for details."
                ),
            }

        logger.info(
            "Map '%s' reviewed and verified: %d → %d tokens (%+d), tool_calls=%d",
            self._active_project, old_tokens, new_tokens,
            new_tokens - old_tokens, tool_calls_used,
        )

        return {
            "updated": True,
            "summary": (
                f"Map updated and verified: {old_tokens} → {new_tokens} tokens "
                f"({new_tokens - old_tokens:+d}). "
                f"{tool_calls_used} tool calls used."
            ),
        }

    async def _create_new_project_map(
        self, project_name: str, summary: str,
    ) -> None:
        """Create a minimal map for a newly detected project.

        Maps are written to the central directory (~/.mesh/memory/maps/),
        so this always proceeds regardless of whether a project_dir exists.
        """
        prompt = NEW_PROJECT_MAP_PROMPT.format(
            project_name=project_name,
            summary=summary,
        )

        try:
            llm_response = await asyncio.wait_for(
                self._llm_client.complete(prompt, max_tokens=4096),
                timeout=60,
            )
        except Exception:
            logger.error(
                "New project map creation failed for '%s'",
                project_name, exc_info=True,
            )
            return

        content = re.sub(r"^```(?:markdown|md)?\s*\n?", "", llm_response.strip())
        content = re.sub(r"\n?```\s*$", "", content).strip()

        if not content.startswith("# Project:"):
            idx = content.find("# Project:")
            if idx >= 0:
                content = content[idx:]

        ok = await self.create_map(project_name, content)
        if ok:
            logger.info(
                "New project map created: '%s' (%d chars)",
                project_name, len(content),
            )
        else:
            logger.warning(
                "Project map bootstrap skipped for '%s': no project_dir known yet",
                project_name,
            )

    def _known_project_names(self) -> str:
        """Get comma-separated list of known project names."""
        if not self._store:
            return "(none)"
        maps = self._store.list_maps()
        if not maps:
            return "(none)"
        return ", ".join(m["project_name"] for m in maps)

    # ── Checkpoint / Recovery ─────────────────────────────────

    def checkpoint_dropped_turns(self, turns: list) -> str | None:
        """Save dropped turns to disk before processing.

        Returns the checkpoint file path, or None on failure.
        """
        import json as json_mod

        if not self._store:
            return None

        db_dir = os.path.dirname(self._store._db_path)
        ckpt_path = os.path.join(db_dir, f"{self._nickname}_checkpoint.json")

        try:
            serialized = []
            for turn in turns:
                ts = getattr(turn, "timestamp", "")
                if hasattr(ts, "isoformat"):
                    ts = ts.isoformat()
                serialized.append({
                    "role": getattr(turn, "role", ""),
                    "content": getattr(turn, "content", ""),
                    "timestamp": str(ts),
                    "from_node": getattr(turn, "from_node", ""),
                    "to_node": getattr(turn, "to_node", None),
                    "meta": getattr(turn, "meta", {}),
                    "seq_id": getattr(turn, "seq_id", 0),
                })

            with open(ckpt_path, "w") as f:
                json_mod.dump(serialized, f)

            logger.info(
                "Checkpoint saved: %d turns to %s", len(turns), ckpt_path,
            )
            return ckpt_path
        except Exception:
            logger.error("Failed to save checkpoint", exc_info=True)
            return None

    def load_checkpoint(self) -> list | None:
        """Load a checkpoint file if it exists. Returns Turn list or None."""
        import json as json_mod

        if not self._store:
            return None

        db_dir = os.path.dirname(self._store._db_path)
        ckpt_path = os.path.join(db_dir, f"{self._nickname}_checkpoint.json")

        if not os.path.exists(ckpt_path):
            return None

        try:
            with open(ckpt_path, "r") as f:
                data = json_mod.load(f)

            # Import Turn here to avoid circular imports at module level
            from mesh.conversation_history import Turn

            turns = []
            for item in data:
                turns.append(Turn(
                    role=item.get("role", ""),
                    content=item.get("content", ""),
                    timestamp=item.get("timestamp", ""),
                    from_node=item.get("from_node", ""),
                    to_node=item.get("to_node"),
                    meta=item.get("meta", {}),
                    seq_id=item.get("seq_id", 0),
                ))

            logger.info("Checkpoint loaded: %d turns from %s", len(turns), ckpt_path)
            return turns
        except Exception:
            logger.error("Failed to load checkpoint", exc_info=True)
            return None

    def clear_checkpoint(self) -> None:
        """Delete the checkpoint file after successful processing."""
        if not self._store:
            return

        db_dir = os.path.dirname(self._store._db_path)
        ckpt_path = os.path.join(db_dir, f"{self._nickname}_checkpoint.json")

        try:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
                logger.info("Checkpoint cleared: %s", ckpt_path)
        except Exception:
            logger.error("Failed to clear checkpoint", exc_info=True)

    async def replay_checkpoint(self) -> bool:
        """On startup, check for and replay any pending checkpoint.

        Returns True if a checkpoint was found and replayed.
        """
        turns = self.load_checkpoint()
        if not turns:
            return False

        logger.info("Replaying checkpoint: %d dropped turns", len(turns))
        try:
            await self.on_window_drop(turns)
            self.clear_checkpoint()
            logger.info("Checkpoint replay complete")
            return True
        except Exception:
            logger.error("Checkpoint replay failed", exc_info=True)
            return False

    # ── Rendering ─────────────────────────────────────────────

    @property
    def conversation_summary(self) -> str:
        """The current conversation summary text (may be empty)."""
        return self._conversation_summary

    async def render_summary_block(self) -> str:
        """Render the conversation summary as XML for prompt injection.

        Placed before the conversation window in the prompt so the LLM
        sees: summary of older context → recent conversation turns.
        """
        if not self._conversation_summary:
            return ""
        return (
            f"<conversation_summary messages_covered=\"{self._summary_messages_summarized}\">\n"
            f"{self._conversation_summary}\n"
            f"</conversation_summary>"
        )

    async def render_maps_block(self) -> str:
        """Render the active project's map as XML (legacy fallback)."""
        if not self._active_project:
            return ""
        content = await self.get_map(self._active_project)
        if not content:
            return ""
        path = self._map_file_path(self._active_project)
        updated = ""
        if path and os.path.exists(path):
            mtime = os.path.getmtime(path)
            updated = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
        return (
            f'<project_map project="{self._active_project}" '
            f'updated="{updated}">\n{content}\n</project_map>'
        )

    async def render_relevant_maps_block(
        self, context_text: str, k: int = 2, min_score: float = 0.2,
    ) -> str:
        """Select and render top-k project maps by relevance to context.

        Falls back to render_maps_block() when no embeddings are available.
        """
        if not context_text.strip():
            return await self.render_maps_block()

        results = await self.select_relevant_maps(context_text, k=k, min_score=min_score)
        if not results:
            return await self.render_maps_block()

        parts: list[str] = []
        for name, score, content in results:
            path = self._map_file_path(name)
            updated = ""
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                updated = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d")
            parts.append(
                f'<project_map project="{name}" relevance="{score:.2f}" '
                f'updated="{updated}">\n{content}\n</project_map>'
            )
        return "\n".join(parts)

    async def render_recent_log_block(self) -> str:
        """Render the last m log entries at reflection depth."""
        if not self._pool:
            return ""
        recent = sorted(self._pool, key=lambda e: e.created_at, reverse=True)
        recent = recent[:self._recent_log_count]
        if not recent:
            return ""

        parts = ["<recent_activity>"]
        for entry in recent:
            date_str = entry.created_at.strftime("%Y-%m-%d")
            parts.append(
                f'<entry date="{date_str}" project="{entry.project}" '
                f'topic="{entry.topic_label}" outcome="{entry.outcome}">'
            )
            # Reflection depth (2-3 paragraphs)
            if entry.reflection:
                parts.append(entry.reflection)
            else:
                parts.append(entry.summary)
            parts.append("</entry>")
        parts.append("</recent_activity>")
        return "\n".join(parts)

    async def render_representative_block(self) -> str:
        """Render FLMI-selected representative memories.

        Draws from full pool (all projects). Project-agnostic.
        Uses existing selection.py for active set maintenance.
        """
        active = self.active_entries
        if not active:
            return ""

        # Sort by weight descending
        active_sorted = sorted(
            active,
            key=lambda e: self._active_weights.get(e.id, 0.0),
            reverse=True,
        )

        parts = ["<memory>"]
        for entry in active_sorted:
            date_str = entry.created_at.strftime("%Y-%m-%d")
            tags_str = ", ".join(entry.tags) if entry.tags else ""
            parts.append(
                f'<entry id="{entry.id}" date="{date_str}" '
                f'tags="{tags_str}" outcome="{entry.outcome}" '
                f'source="representative">'
            )
            parts.append(entry.summary)
            parts.append("</entry>")
        parts.append("</memory>")
        return "\n".join(parts)

    async def render_retrieved_context(
        self, query: str, budget_tokens: int
    ) -> str:
        """Search log entries and render results for injection.

        Phase 5 will wire this into the router classification loop.
        """
        query_emb = await self._get_query_embedding(query)
        if query_emb is None:
            logger.warning("Retrieval failed: could not embed query='%s'", query[:100])
            return ""

        scored = []
        for entry in self._pool:
            if entry.retrieval_key_embedding is None:
                continue
            sim = cosine_sim(query_emb, entry.retrieval_key_embedding)
            scored.append((sim, entry))
        scored.sort(key=lambda x: x[0], reverse=True)

        parts = []
        from mesh.llm import estimate_tokens
        token_count = 0
        for sim, entry in scored[:self._retrieval_k]:
            text = entry.reflection if entry.reflection else entry.summary
            entry_tokens = estimate_tokens(text) + 30  # XML overhead
            if token_count + entry_tokens > budget_tokens:
                break
            date_str = entry.created_at.strftime("%Y-%m-%d")
            parts.append(
                f'<log_entry date="{date_str}" topic="{entry.topic_label}" '
                f'outcome="{entry.outcome}" similarity="{sim:.2f}">\n'
                f'{text}\n</log_entry>'
            )
            token_count += entry_tokens

        if not parts:
            logger.warning("Retrieval returned 0 results for query='%s'", query[:100])
            return ""

        logger.info(
            "Retrieved %d entries (%d tokens) for query='%s'",
            len(parts), token_count, query[:100],
        )
        return "\n".join(parts)

    # ── Personality (carried from v1) ─────────────────────────

    def get_personality(self) -> str:
        return self._personality_cache

    def set_personality(self, text: str) -> None:
        self._personality_cache = text
        if self._store is not None:
            self._store.set_personality(text)

    def seed_personality(self, config_text: str) -> None:
        if not config_text or self._store is None:
            return
        if not self._personality_cache:
            self._store.set_personality(config_text)
            self._personality_cache = config_text
            logger.info("Personality seeded from config for '%s'", self._nickname)

    # ── FLMI Active Set (retained from v1 for representative block) ──

    def _reselect_active_set(self) -> None:
        """Run full greedy submodular selection over the pool."""
        if not self._pool:
            self._active_ids = set()
            self._active_weights = {}
            self._active_f_s = 0.0
            return

        embs = [e.reflection_embedding for e in self._pool
                if e.reflection_embedding is not None]
        emb_to_pool = [i for i, e in enumerate(self._pool)
                       if e.reflection_embedding is not None]

        if not embs:
            self._active_ids = set()
            self._active_weights = {}
            self._active_f_s = 0.0
            return

        selected_indices, weights, f_s = select_active_set(
            embs, self._active_size
        )

        self._active_ids = set()
        self._active_weights = {}
        for sel_pos, emb_idx in enumerate(selected_indices):
            pool_idx = emb_to_pool[emb_idx]
            entry = self._pool[pool_idx]
            self._active_ids.add(entry.id)
            self._active_weights[entry.id] = weights[sel_pos]
            entry.weight = weights[sel_pos]
        self._active_f_s = f_s

        for e in self._pool:
            if e.id not in self._active_ids:
                e.weight = 0.0

        if self._store:
            self._store.update_weights_batch(
                {e.id: e.weight for e in self._pool}
            )

    def _incremental_active_update(self, new_entry: MemoryEntry) -> bool:
        """Incrementally update active set when a new entry is added."""
        active = self.active_entries
        active_with_embs = [e for e in active
                            if e.reflection_embedding is not None]
        active_embs = [e.reflection_embedding for e in active_with_embs]

        if new_entry.reflection_embedding is None:
            return False

        accepted, evict_idx, new_weights, new_f_s = try_swap(
            active_embs,
            new_entry.reflection_embedding,
            self._active_size,
            cached_weights=[self._active_weights.get(e.id, 0.0)
                            for e in active_with_embs],
            cached_f_s=self._active_f_s,
        )

        if accepted:
            if evict_idx is not None:
                evicted = active_with_embs[evict_idx]
                self._active_ids.discard(evicted.id)
                del self._active_weights[evicted.id]
                evicted.weight = 0.0

            self._active_ids.add(new_entry.id)
            updated_active = self.active_entries
            self._active_weights = {}
            for i, e in enumerate(updated_active):
                e.weight = new_weights[i]
                self._active_weights[e.id] = new_weights[i]
            self._active_f_s = new_f_s

            if self._store:
                self._store.update_weights_batch(
                    {e.id: e.weight for e in self._pool}
                )
            return True
        return False

    def _prune_pool(self) -> int:
        """Prune oldest pool-only entries when over cap."""
        if len(self._pool) <= self._pool_max_entries:
            return 0
        excess = len(self._pool) - self._pool_max_entries
        pool_only = sorted(
            [e for e in self._pool if e.id not in self._active_ids],
            key=lambda e: e.created_at,
        )
        to_remove = pool_only[:excess]
        if not to_remove:
            return 0
        remove_ids = {e.id for e in to_remove}
        self._pool = [e for e in self._pool if e.id not in remove_ids]
        for entry_id in remove_ids:
            self._store.delete(entry_id)
        logger.info(
            "Memory pool pruned: removed %d entries (pool size: %d)",
            len(remove_ids), len(self._pool),
        )
        return len(remove_ids)

    # ── Retrieval helpers ─────────────────────────────────────

    async def _get_query_embedding(self, query: str) -> np.ndarray | None:
        cache_key = query
        if cache_key in self._query_embedding_cache:
            return self._query_embedding_cache[cache_key]
        try:
            emb = await self._embedder.embed_to_array(query)
            self._query_embedding_cache[cache_key] = emb
            if len(self._query_embedding_cache) > 10:
                oldest = next(iter(self._query_embedding_cache))
                del self._query_embedding_cache[oldest]
            return emb
        except Exception:
            logger.error("Failed to embed query", exc_info=True)
            return None

    # ── v1-compatible API surface ─────────────────────────────
    # These methods allow MemorySystemV2 to be used as a drop-in
    # replacement for MemorySystem in agent_node.py and
    # tool_implementations.py. v1 rendering is NOT supported —
    # callers should check memory_version before calling render().

    def should_reflect(self, result, stats=None) -> bool:
        """v1 compat: significance gate (used by v1 code paths)."""
        now = time.monotonic()
        if (now - self._last_reflection_time) < self._reflection_cooldown_secs:
            return False
        if stats is not None:
            return self._should_reflect_from_stats(stats)
        return self._should_reflect_legacy(result)

    def _should_reflect_from_stats(self, stats) -> bool:
        if stats.tool_calls >= self._reflection_min_tools:
            return True
        if (stats.num_user_visible_turns >= self._reflection_min_discussion_turns
                and stats.total_user_visible_chars >= self._reflection_min_discussion_chars):
            return True
        if (stats.tool_calls <= self._reflection_max_brainstorm_tools
                and stats.agent_response_chars >= self._reflection_min_brainstorm_response_chars
                and stats.num_user_visible_turns >= 2):
            return True
        if stats.has_errors:
            return True
        return False

    def _should_reflect_legacy(self, result) -> bool:
        tool_count = 0
        has_errors = False
        for entry in getattr(result, "context", []):
            msg = entry.message if hasattr(entry, "message") else entry
            metadata = getattr(msg, "metadata", {}) or {}
            if metadata.get("tool_calls"):
                tool_count += 1
            if metadata.get("tool_results"):
                content = getattr(msg, "content", "")
                if isinstance(content, str) and "error" in content.lower():
                    has_errors = True
        if result.error is not None:
            has_errors = True
        if tool_count >= self._reflection_min_tools:
            return True
        if has_errors:
            return True
        return False

    async def reflect_on_completion(
        self, trigger: str, result, worker_id: str, topic_label: str = "",
    ) -> None:
        """v1 compat: episode-triggered reflection."""
        # Delegate to v1 reflection logic (same as MemorySystem)
        from .system import REFLECTION_PROMPT
        try:
            self._last_reflection_time = time.monotonic()
            trace = self._build_trace(result)
            outcome_text = result.response or "(no response)"
            prompt = REFLECTION_PROMPT.format(
                trigger=trigger,
                trace=trace,
                outcome=outcome_text[:2000],
            )
            llm_response = await asyncio.wait_for(
                self._llm_client.complete(prompt, max_tokens=16384),
                timeout=180,
            )
            reflection = _extract_tag(llm_response, "reflection")
            summary = _extract_tag(llm_response, "summary")
            tags_str = _extract_tag(llm_response, "tags")
            outcome_label = _extract_tag(llm_response, "outcome_label")
            retrieval_key = _extract_tag(llm_response, "retrieval_key")
            project = _extract_tag(llm_response, "project")

            if not reflection or not summary:
                logger.warning("Memory reflection produced empty output")
                return

            if not retrieval_key:
                retrieval_key = summary

            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
            outcome = outcome_label.strip() if outcome_label else "success"
            if outcome not in ("success", "partial", "failure"):
                outcome = "success"

            emb_results = await asyncio.wait_for(
                self._embedder.embed_batch_to_arrays([reflection, retrieval_key]),
                timeout=30,
            )

            entry = MemoryEntry(
                id=MemoryEntry.new_id(),
                created_at=datetime.now(timezone.utc),
                summary=summary,
                reflection=reflection,
                trace=trace,
                trigger=trigger,
                retrieval_key=retrieval_key,
                topic_label=topic_label,
                tags=tags,
                outcome=outcome,
                reflection_embedding=emb_results[0],
                retrieval_key_embedding=emb_results[1],
                weight=0.0,
                project=project or (self._active_project or ""),
            )

            self._pool.append(entry)
            self._store.insert(entry)
            self._incremental_active_update(entry)
            self._prune_pool()

        except asyncio.TimeoutError:
            logger.warning("Memory reflection timed out")
        except Exception:
            logger.error("Memory reflection failed", exc_info=True)

    def _build_trace(self, result) -> str:
        """Construct clipped trace from worker result context.

        Prefers tool_calls/tool_results metadata when present (worker-mode
        results carry these). Falls back to formatting raw routed Message
        content as USER/AGENT lines so topic-segment-flush results — whose
        context is raw `Message` objects without tool metadata — still
        produce a non-empty trace.
        """
        lines = []
        char_budget = self._trace_max_tokens * 4
        saw_tool_metadata = False

        for entry in getattr(result, "context", []):
            msg = entry.message if hasattr(entry, "message") else entry
            metadata = getattr(msg, "metadata", {}) or {}
            content = getattr(msg, "content", "")

            if metadata.get("tool_calls"):
                saw_tool_metadata = True
                calls = metadata["tool_calls"]
                if isinstance(calls, list):
                    for call in calls:
                        if isinstance(call, dict):
                            name = call.get("name", call.get("tool_name", "?"))
                            args = str(call.get("arguments", call.get("args", "")))
                            lines.append(f"[TOOL] {name}({args[:200]})")
                elif isinstance(calls, str):
                    lines.append(f"[TOOL] {calls[:300]}")
            elif metadata.get("tool_results"):
                saw_tool_metadata = True
                result_str = str(content)[:500]
                lines.append(f"[RESULT] {result_str}")

        # Fallback: if no tool metadata was found, format messages as text.
        # This is the dominant case for topic-segment-flush results.
        if not saw_tool_metadata:
            lines = []
            for entry in getattr(result, "context", []):
                msg = entry.message if hasattr(entry, "message") else entry
                content = getattr(msg, "content", "")
                content_str = content if isinstance(content, str) else str(content)
                if not content_str.strip():
                    continue
                from_node = getattr(msg, "from_node", "") or ""
                if from_node.startswith("user:"):
                    prefix = "USER"
                elif from_node.startswith("agent:"):
                    prefix = "AGENT"
                elif from_node.startswith("channel:"):
                    prefix = f"CHANNEL {from_node[len('channel:'):][:30]}"
                else:
                    prefix = f"MSG from={from_node}" if from_node else "MSG"
                lines.append(f"[{prefix}] {content_str[:1000]}")

        trace = "\n".join(lines)
        if len(trace) > char_budget:
            trace = trace[:char_budget] + "\n... (trace truncated)"
        return trace

    # v1-compat rendering (delegates to old three-slice for v1 callers)
    async def render_block(self, query: str | None = None) -> str:
        """Legacy render for router — returns representative block for v2."""
        return await self.render_representative_block()

    async def render_block_for_query(
        self, query: str, k: int | None = None,
        n_full_reflections: int | None = None, tag: str | None = None,
    ) -> str:
        """Legacy render for worker — returns recent log block for v2."""
        return await self.render_recent_log_block()

    async def search_block(
        self,
        query: str,
        k: int = 5,
        project: str | None = None,
        tag: str | None = None,
        mode: str = "hybrid",
    ) -> str:
        """Search memory pool returning a rendered block of top-k entries.

        mode:
          "embedding" — cosine similarity only (original behavior)
          "lexical"   — FTS5/BM25 full-text search only
          "hybrid"    — reciprocal-rank fusion of embedding + lexical

        project semantics:
          None → active project (default)
          ""   → no project filter (all projects)
          str  → that specific project
        """
        if project is None:
            project = self._active_project
        elif project == "":
            project = None

        if project:
            candidates = self._store.list_by_project(project, include_project_empty=True)
        else:
            candidates = self._pool[:]

        if tag:
            candidates = [e for e in candidates if tag in e.tags]

        candidate_ids = {e.id for e in candidates}
        entry_by_id = {e.id: e for e in candidates}

        # ── Embedding ranking ──
        embedding_ranking: list[tuple[str, float]] = []
        if mode in ("embedding", "hybrid"):
            q_emb = await self._get_query_embedding(query)
            if q_emb is None and mode == "embedding":
                return "Search failed: could not embed query."
            if q_emb is not None:
                scored = []
                for e in candidates:
                    emb = e.reflection_embedding if e.reflection_embedding is not None else e.retrieval_key_embedding
                    if emb is None:
                        continue
                    sim = cosine_sim(q_emb, emb)
                    scored.append((e.id, sim))
                scored.sort(key=lambda x: x[1], reverse=True)
                embedding_ranking = scored

        # ── Lexical ranking (FTS5/BM25) ──
        lexical_ranking: list[tuple[str, float]] = []
        if mode in ("lexical", "hybrid"):
            fts_results = self._store.search_fts(query, limit=100)
            lexical_ranking = [
                (eid, score) for eid, score in fts_results
                if eid in candidate_ids
            ]

        # ── Merge rankings ──
        if mode == "hybrid":
            final_ranking = _reciprocal_rank_fusion(
                [embedding_ranking, lexical_ranking]
            )
        elif mode == "lexical":
            final_ranking = [(eid, score) for eid, score in lexical_ranking]
        else:
            final_ranking = [(eid, score) for eid, score in embedding_ranking]

        if not final_ranking:
            return ""

        parts = [f'<memory_search_results query="{query[:80]}" k="{k}" mode="{mode}">']
        for eid, score in final_ranking[:k]:
            e = entry_by_id.get(eid)
            if e is None:
                continue
            date_str = e.created_at.strftime("%Y-%m-%d")
            parts.append(
                f'<entry id="{e.id}" date="{date_str}" project="{e.project}" '
                f'score="{score:.4f}">'
            )
            parts.append(f"**ID**: {e.id}")
            parts.append(f"**Retrieval key**: {e.retrieval_key or '(none)'}")
            parts.append(f"**Summary**: {e.summary}")
            parts.append("</entry>")
        parts.append("</memory_search_results>")
        return "\n".join(parts)

    # v1-compat entry management
    def remember(self, entry_id: str, full: bool = False) -> str | None:
        entry = self._find_entry(entry_id)
        if entry is None:
            return None
        if full:
            return f"## Reflection\n\n{entry.reflection}\n\n## Trace\n\n{entry.trace}"
        return entry.reflection

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        return self._find_entry(entry_id)

    def list_entries(self) -> list[MemoryEntry]:
        return list(self._pool)

    def is_active(self, entry_id: str) -> bool:
        return entry_id in self._active_ids

    async def add_entry(
        self, summary: str, reflection: str = "", trace: str = "",
        tags: list[str] | None = None, outcome: str = "success",
    ) -> tuple[MemoryEntry, bool]:
        """Manually add a memory entry."""
        emb_texts = [reflection or summary, summary]
        embs = await asyncio.wait_for(
            self._embedder.embed_batch_to_arrays(emb_texts),
            timeout=30,
        )
        entry = MemoryEntry(
            id=MemoryEntry.new_id(),
            created_at=datetime.now(timezone.utc),
            summary=summary,
            reflection=reflection or summary,
            trace=trace,
            trigger=summary,
            retrieval_key=summary,
            tags=tags or [],
            outcome=outcome,
            reflection_embedding=embs[0],
            retrieval_key_embedding=embs[1],
            weight=0.0,
            project=self._active_project or "",
        )
        self._pool.append(entry)
        self._store.insert(entry)
        in_active = self._incremental_active_update(entry)
        self._prune_pool()
        return entry, in_active

    async def delete_entry(self, entry_id: str) -> bool:
        idx = None
        for i, e in enumerate(self._pool):
            if e.id == entry_id:
                idx = i
                break
        if idx is None:
            return False
        was_active = entry_id in self._active_ids
        self._pool.pop(idx)
        self._store.delete(entry_id)
        if was_active:
            self._active_ids.discard(entry_id)
            if entry_id in self._active_weights:
                del self._active_weights[entry_id]
            self._reselect_active_set()
        return True

    async def edit_entry(
        self,
        entry_id: str,
        *,
        summary: str | None = None,
        reflection: str | None = None,
        retrieval_key: str | None = None,
        tags: list[str] | None = None,
        outcome: str | None = None,
    ) -> str:
        """Edit a memory entry in place, keeping the ID stable.

        Re-embeds the changed fields so search stays consistent.
        Returns a confirmation or error message.
        """
        entry = self._find_entry(entry_id)
        if entry is None:
            return f"Error: no memory entry with ID '{entry_id}'."

        before_summary = entry.summary
        before_retrieval_key = entry.retrieval_key

        if summary is not None:
            entry.summary = summary
        if reflection is not None:
            entry.reflection = reflection
        if retrieval_key is not None:
            entry.retrieval_key = retrieval_key
        if tags is not None:
            entry.tags = tags
        if outcome is not None:
            entry.outcome = outcome

        reembed = (
            (summary is not None and summary != before_summary)
            or (retrieval_key is not None and retrieval_key != before_retrieval_key)
            or (reflection is not None)
        )
        if reembed:
            try:
                emb_texts = [
                    entry.reflection or entry.summary,
                    entry.retrieval_key or entry.summary,
                ]
                embs = await asyncio.wait_for(
                    self._embedder.embed_batch_to_arrays(emb_texts),
                    timeout=30,
                )
                entry.reflection_embedding = embs[0]
                entry.retrieval_key_embedding = embs[1]
            except Exception:
                logger.warning("Re-embedding failed for entry %s; keeping old embeddings", entry_id)

        from .store import _SENTINEL
        self._store.update_entry(
            entry_id,
            summary=summary,
            reflection=reflection,
            retrieval_key=retrieval_key,
            tags=tags,
            outcome=outcome,
            reflection_embedding=entry.reflection_embedding if reembed else _SENTINEL,
            retrieval_key_embedding=entry.retrieval_key_embedding if reembed else _SENTINEL,
        )
        return f"Memory entry '{entry_id}' updated successfully."

    def _find_entry(self, entry_id: str) -> MemoryEntry | None:
        for e in self._pool:
            if e.id == entry_id:
                return e
        return None

    # ── Diagnostics ───────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        # Get active map size if available
        active_map_chars = 0
        if self._active_project:
            path = self._map_file_path(self._active_project)
            if path and os.path.exists(path):
                try:
                    active_map_chars = os.path.getsize(path)
                except OSError:
                    pass

        return {
            "version": 2,
            "pool_size": len(self._pool),
            "active_set_size": len(self._active_ids),
            "active_set_target": self._active_size,
            "pool_max_entries": self._pool_max_entries,
            "active_project": self._active_project,
            "active_map_chars": active_map_chars,
            "map_count": len(self._store.list_maps()) if self._store else 0,
            "last_reflection_ago_seconds": (
                round(time.monotonic() - self._last_reflection_time, 1)
                if self._last_reflection_time > 0 else None
            ),
            "reflection_cooldown_seconds": self._reflection_cooldown_secs,
            "retrieval_k": self._retrieval_k,
        }

    # ── Memory Formation v3 (see docs/plans/memory-formation-v3-2026-04-27.md) ──

    def get_formation_cursor(self) -> tuple[int, str]:
        """Return (last_index, last_ts_utc). (0, "") if uninitialized."""
        if not self._store:
            return (0, "")
        return self._store.get_formation_cursor()

    async def form_un_formed(
        self, history: list, reason: str,
    ) -> int:
        """Form everything from cursor forward, then advance cursor.

        Returns number of memory entries created. `reason` is one of
        {"time-based", "token-pressure", "shutdown", "startup"} — used
        for logging/metrics. See plan §2.7.3.
        """
        if not self._formation_v3_enabled:
            return 0
        if not self._store:
            logger.warning("form_un_formed called before store initialized")
            return 0
        if self._formation_lock is None:
            self._formation_lock = asyncio.Lock()

        async with self._formation_lock:
            cursor_idx, _ = self._store.get_formation_cursor()
            if cursor_idx > len(history):
                logger.warning(
                    "v3 formation: cursor %d > history length %d "
                    "(likely agent restart with shorter window) — "
                    "clamping cursor to 0",
                    cursor_idx, len(history),
                )
                cursor_idx = 0
                self._store.set_formation_cursor(0, "")
            if cursor_idx >= len(history):
                return 0

            un_formed = list(history[cursor_idx:])
            if not un_formed:
                return 0

            window_key = (cursor_idx, cursor_idx + len(un_formed))
            prev_failures = self._parse_failure_count.get(window_key, 0)

            try:
                seg_result = await self._run_formation_v3(
                    un_formed, base_offset=cursor_idx,
                )
            except Exception as e:
                seg_result = e

            new_cursor = cursor_idx + len(un_formed)
            new_ts_utc = _turn_ts_iso(un_formed[-1])

            n_entries = 0
            new_entries: list[MemoryEntry] = []
            cursor_advanced = False

            # Capture either segmenter or persist failure as a single
            # "advance was prevented" category so the 3-strike fallback covers
            # both (rev-7 fix to code-review Issue 1).
            failure: Exception | None = None
            failure_kind = ""
            if isinstance(seg_result, Exception):
                failure = seg_result
                failure_kind = "segmenter"
            else:
                try:
                    n_entries, new_entries = await self._persist_v3_entries_atomic(
                        segments=seg_result,
                        new_cursor=new_cursor,
                        new_ts_utc=new_ts_utc,
                    )
                    cursor_advanced = True
                except Exception as e:
                    failure = e
                    failure_kind = "persist"
                    n_entries = 0
                    new_entries = []

            if failure is not None:
                logger.warning(
                    "v3 formation %s failed (%s): %s",
                    failure_kind, reason, failure,
                )
                failures = prev_failures + 1
                self._parse_failure_count[window_key] = failures
                if failures >= self._parse_failure_fallback_threshold:
                    logger.error(
                        "v3 formation: %d consecutive %s failures on window %s — "
                        "writing fallback entry",
                        failures, failure_kind, window_key,
                    )
                    n_entries, new_entries = await self._write_parse_failure_fallback(
                        un_formed=un_formed,
                        new_cursor=new_cursor,
                        new_ts_utc=new_ts_utc,
                    )
                    self._parse_failure_count.pop(window_key, None)
                    cursor_advanced = True
                    # Reset agent-node token counter (rev 6 fallback path).
                    self._notify_cursor_advance()
                # else: cursor stays put — retry next trigger.
            else:
                self._parse_failure_count.pop(window_key, None)
                # Reset agent-node token counter on cursor advance (§2.7.9).
                self._notify_cursor_advance()

            # New-project bootstrap (rev 5 / Alice Issue 1).
            for entry in new_entries:
                if entry.project and entry.project != (self._active_project or ""):
                    existing = self._store.get_map(entry.project)
                    if not existing:
                        logger.info(
                            "New project detected (v3): '%s' — bootstrapping map",
                            entry.project,
                        )
                        try:
                            await self._create_new_project_map(
                                entry.project, entry.summary,
                            )
                        except Exception as e:
                            logger.warning(
                                "v3 new-project bootstrap failed for '%s': %s",
                                entry.project, e,
                            )

            # Fire-and-forget: integrate entries into project maps.
            if new_entries:
                asyncio.create_task(
                    self._integrate_entries_into_maps(new_entries)
                )

            logger.info(
                "v3 formation complete (%s): %d entries from %d turns "
                "[cursor %d → %d]",
                reason, n_entries, len(un_formed),
                cursor_idx, new_cursor if cursor_advanced else cursor_idx,
            )
            return n_entries

    def _notify_cursor_advance(self) -> None:
        """Invoke the cursor-advance callback (set by AgentNode for token counter reset)."""
        cb = self._on_cursor_advance
        if cb is None:
            return
        try:
            cb()
        except Exception as e:
            logger.warning("on_cursor_advance callback raised: %s", e)

    async def _run_formation_v3(
        self, turns: list, *, base_offset: int = 0,
    ):
        """Run the LLMSegmenterV3 on a list of turns.

        Returns the list of Segment objects. The base_offset is informational
        for callers that want global indices later; segments already carry
        their global indices because we pass the full `turns` slice.
        """
        from .formation_v3 import LLMSegmenterV3

        segmenter = LLMSegmenterV3(
            self._formation_llm_client or self._llm_client,
            window_size=self._formation_v3_window_size,
            overlap=self._formation_v3_overlap,
            defer_tail_turns=self._formation_v3_defer_tail,
            model=self._formation_v3_model,
        )
        # Constrain the segmenter to known project names to prevent
        # hallucinated identifiers (e.g. "class-sp26" vs real "sp26-221").
        known_projects = self._known_project_names()
        return await segmenter.segment(turns, known_projects=known_projects)

    async def _persist_v3_entries_atomic(
        self,
        segments: list,
        new_cursor: int,
        new_ts_utc: str,
    ) -> tuple[int, list[MemoryEntry]]:
        """Persist worthwhile segments + advance cursor in one transaction."""
        worthwhile = [
            s for s in segments
            if s.metadata.get("worthwhile", False)
        ]

        if not worthwhile:
            # No memory entries; still advance cursor (the un-formed range
            # was processed — segmenter found nothing worth persisting).
            self._store.set_formation_cursor(new_cursor, new_ts_utc)
            return 0, []

        # Compute embeddings in batch (cheaper than per-entry).
        emb_targets_reflection: list[str] = []
        emb_targets_retrieval: list[str] = []
        for seg in worthwhile:
            md = seg.metadata
            summary = md.get("summary", "") or ""
            retrieval_key = md.get("retrieval_key", "") or ""
            target = (summary + " " + retrieval_key).strip() or summary or retrieval_key
            emb_targets_reflection.append(target)
            emb_targets_retrieval.append(retrieval_key)

        try:
            reflection_embs = await asyncio.wait_for(
                self._embedder.embed_batch_to_arrays(emb_targets_reflection),
                timeout=60,
            )
        except Exception as e:
            logger.warning("v3 reflection embedding failed: %s", e)
            reflection_embs = [None] * len(worthwhile)

        try:
            retrieval_embs = await asyncio.wait_for(
                self._embedder.embed_batch_to_arrays(emb_targets_retrieval),
                timeout=60,
            )
        except Exception as e:
            logger.warning("v3 retrieval embedding failed: %s", e)
            retrieval_embs = [None] * len(worthwhile)

        new_entries: list[MemoryEntry] = []
        for i, seg in enumerate(worthwhile):
            md = seg.metadata
            seg_turns = seg.turns
            first_user_text = ""
            for t in seg_turns:
                role = getattr(t, "role", "")
                from_node = getattr(t, "from_node", "")
                if role == "user" or from_node.startswith("user:"):
                    first_user_text = str(getattr(t, "content", ""))
                    break

            outcome = md.get("outcome") or "partial"
            if outcome not in ("success", "partial", "failure"):
                outcome = "partial"

            score = int(md.get("score", 0))
            score = max(0, min(10, score))
            tags = list(md.get("tags") or [])
            tags.append(f"score:{score}")

            raw_project = md.get("project")
            project = raw_project if raw_project else ""

            topic_label = md.get("topic_label", "") or seg.topic_label or "untitled"
            trigger = f"[TOPIC: {topic_label}] {first_user_text[:500]}"

            entry = MemoryEntry(
                id=MemoryEntry.new_id(),
                created_at=datetime.now(timezone.utc),
                summary=md.get("summary", "") or "",
                reflection="",
                trace=self._format_turns_as_trace(seg_turns)[: self._trace_max_tokens * 4],
                trigger=trigger,
                retrieval_key=md.get("retrieval_key", "") or "",
                topic_label=topic_label,
                tags=tags,
                outcome=outcome,
                project=project,
                reflection_embedding=reflection_embs[i],
                retrieval_key_embedding=retrieval_embs[i],
                weight=0.0,
            )
            new_entries.append(entry)

        # Atomic insert + cursor advance.
        self._store.insert_entry_and_advance_cursor(
            entries=new_entries,
            new_cursor=new_cursor,
            new_ts_utc=new_ts_utc,
        )

        # In-memory pool / FLMI bookkeeping.
        for entry in new_entries:
            self._pool.append(entry)
        if len(new_entries) > 1:
            self._reselect_active_set()
        elif new_entries:
            self._incremental_active_update(new_entries[0])
        self._prune_pool()

        return len(new_entries), new_entries

    async def _write_parse_failure_fallback(
        self, un_formed: list, new_cursor: int, new_ts_utc: str,
    ) -> tuple[int, list[MemoryEntry]]:
        """Write a placeholder entry tagged `formation-fallback` and advance cursor."""
        first_user_text = ""
        for t in un_formed:
            role = getattr(t, "role", "")
            from_node = getattr(t, "from_node", "")
            if role == "user" or from_node.startswith("user:"):
                first_user_text = str(getattr(t, "content", ""))[:200]
                break

        n_turns = len(un_formed)
        placeholder = MemoryEntry(
            id=MemoryEntry.new_id(),
            created_at=datetime.now(timezone.utc),
            summary=f"(formation parse failure on {n_turns} turns)",
            reflection="",
            trace=self._format_turns_as_trace(un_formed)[: self._trace_max_tokens * 4],
            trigger="[formation-fallback]",
            retrieval_key=first_user_text or "(no user text in window)",
            topic_label="formation-fallback",
            tags=["formation-fallback", "score:0", f"turns:{n_turns}"],
            outcome="partial",
            project=self._active_project or "",
            reflection_embedding=None,
            retrieval_key_embedding=None,
            weight=0.0,
        )

        self._store.insert_entry_and_advance_cursor(
            entries=[placeholder],
            new_cursor=new_cursor,
            new_ts_utc=new_ts_utc,
        )
        self._pool.append(placeholder)
        return 1, [placeholder]

    async def _maybe_run_v3_embedding_migration(self) -> None:
        """Re-embed reflection_embedding on summary+retrieval_key (§2.4.1).

        Idempotent: tracked in `migrations_complete` table. Runs once per
        agent on first v3 boot.
        """
        if not self._formation_v3_enabled:
            return
        if not self._store:
            return
        if self._store.is_migration_complete("v3_reflection_embedding"):
            return

        entries = self._store.load()
        logger.info(
            "v3 embedding migration starting: %d entries", len(entries),
        )

        # Build target list, preserving entry IDs.
        targets: list[str] = []
        ids: list[str] = []
        for e in entries:
            target = ((e.summary or "") + " " + (e.retrieval_key or "")).strip()
            if not target:
                continue
            targets.append(target)
            ids.append(e.id)

        if not targets:
            self._store.mark_migration_complete("v3_reflection_embedding")
            return

        try:
            new_embs = await asyncio.wait_for(
                self._embedder.embed_batch_to_arrays(targets),
                timeout=120,
            )
        except Exception as e:
            logger.error(
                "v3 embedding migration aborted (will retry next boot): %s", e,
            )
            return

        # Atomic bulk update: every entry's embedding plus the migration marker
        # commit together, or none of them do (rev-7 fix to code-review Issue 2).
        try:
            self._store.bulk_update_reflection_embeddings_and_mark_migration(
                list(zip(ids, new_embs)),
                "v3_reflection_embedding",
            )
        except Exception as e:
            logger.error(
                "v3 embedding migration transaction failed (will retry next boot): %s",
                e,
            )
            return

        # Refresh in-memory pool to reflect new embeddings.
        self._pool = self._store.load()
        self._reselect_active_set()

        logger.info(
            "v3 embedding migration complete: %d entries re-embedded",
            len(ids),
        )

    # ── Memory retrieval redesign: TOC builder ─────────────────

    async def build_toc(
        self,
        query_text: str | None = None,
        k: int = 30,
        project: str | None = None,
        ranking: str = "cosine",
        context_text: str | None = None,
        restriction_size: int = 100,
    ) -> list[TocEntry]:
        """Build the auto-injected memory table of contents.

        Two-stage pipeline when embeddings are available:
          Stage 1 — Relevance restriction: embed context, cosine filter to
                    top ``restriction_size`` candidates.
          Stage 2 — FL diversity: run lazy greedy Facility Location on the
                    restricted set to select ``k`` diverse items.

        Falls back to pure-cosine ranking when the pool is small, no
        embeddings exist, or the embedder is unavailable.

        Args:
            query_text: query for relevance ranking (latest user message).
            k: max TOC entries to return.
            project: project filter; None -> no filter (all projects).
                Pass a project name to scope to that project.
            ranking: "cosine" | "hybrid" (hybrid adds recency/score bonuses).
            context_text: broader conversation context (last N turns) for
                embedding. Falls back to ``query_text`` if not provided.
            restriction_size: how many candidates survive Stage 1.
        """
        # project=None → full pool (all projects). Explicit name → scoped.
        if project:
            candidates = self._store.list_by_project(
                project, include_project_empty=True,
            )
        else:
            candidates = self._pool[:]

        if not candidates:
            return []

        # Determine the embedding query — prefer broader context
        embed_text = context_text or query_text

        if embed_text:
            q_emb = await self._get_query_embedding(embed_text)
        else:
            q_emb = None

        if q_emb is not None:
            # ── Stage 1: relevance restriction ──
            scored: list[tuple[float, MemoryEntry]] = []
            for entry in candidates:
                emb = entry.retrieval_key_embedding
                if emb is None:
                    emb = entry.reflection_embedding
                if emb is None:
                    sim = -1.0
                else:
                    sim = float(cosine_sim(q_emb, emb))
                if ranking == "hybrid":
                    age_days = (datetime.now(timezone.utc) - entry.created_at).days
                    recency = max(0.0, 1.0 - age_days / 30.0) * 0.05
                    score_tag = next(
                        (int(t.split(":")[1]) for t in entry.tags
                         if t.startswith("score:") and t.split(":")[1].isdigit()),
                        5,
                    )
                    sim += recency + (score_tag - 5) * 0.01
                scored.append((sim, entry))
            scored.sort(key=lambda x: x[0], reverse=True)

            restricted = scored[:restriction_size]
            logger.info(
                "TOC Stage 1: %d candidates → top %d (scores %.3f–%.3f)",
                len(scored), len(restricted),
                restricted[-1][0] if restricted else 0.0,
                restricted[0][0] if restricted else 0.0,
            )

            # ── Stage 2: FL diversity selection via lazy greedy ──
            if len(restricted) > k:
                embs: list[np.ndarray] = []
                for _sim, entry in restricted:
                    emb = entry.retrieval_key_embedding
                    if emb is None:
                        emb = entry.reflection_embedding
                    if emb is not None:
                        embs.append(emb)
                    else:
                        embs.append(np.zeros(q_emb.shape[0], dtype=np.float32))

                sim_matrix = _build_sim_matrix(embs)
                fl_indices = lazy_greedy_fl(sim_matrix, k)
                out: list[TocEntry] = []
                for idx in fl_indices:
                    rel_score, entry = restricted[idx]
                    rkey = entry.retrieval_key or entry.summary[:150]
                    out.append(TocEntry(
                        id=entry.id,
                        retrieval_key=rkey,
                        project=entry.project,
                        tags=list(entry.tags),
                        score=float(rel_score),
                    ))
                out.sort(key=lambda e: e.score, reverse=True)

                projects = sorted({e.project for e in out if e.project})
                logger.info(
                    "TOC Stage 2: FL selected %d from %d | ids=[%s] projects=[%s]",
                    len(fl_indices), len(restricted),
                    ",".join(e.id for e in out),
                    ",".join(projects),
                )
                return out
            else:
                # Pool too small for FL — just return the restricted set
                out = []
                for score, entry in restricted[:k]:
                    rkey = entry.retrieval_key or entry.summary[:150]
                    out.append(TocEntry(
                        id=entry.id,
                        retrieval_key=rkey,
                        project=entry.project,
                        tags=list(entry.tags),
                        score=float(score),
                    ))
                projects = sorted({e.project for e in out if e.project})
                logger.info(
                    "TOC selected (no FL): %d entries | ids=[%s] projects=[%s]",
                    len(out),
                    ",".join(e.id for e in out),
                    ",".join(projects),
                )
                return out
        else:
            # No embedding — recency fallback
            logger.info("TOC: no embedding available — recency fallback (%d candidates)", len(candidates))
            recency_scored = [
                (entry.created_at.timestamp(), entry)
                for entry in candidates
            ]
            recency_scored.sort(key=lambda x: x[0], reverse=True)
            out = []
            for score, entry in recency_scored[:k]:
                rkey = entry.retrieval_key or entry.summary[:150]
                out.append(TocEntry(
                    id=entry.id,
                    retrieval_key=rkey,
                    project=entry.project,
                    tags=list(entry.tags),
                    score=float(score),
                ))
            return out

    def dedup_toc_against_window(
        self,
        toc: list[TocEntry],
        conv_history,
    ) -> list[TocEntry]:
        """Mark TOC entries already present in the conversation window."""
        if conv_history is None:
            return toc

        already: dict[str, bool] = {}

        if hasattr(conv_history, "iter_tool_results"):
            for t in conv_history.iter_tool_results(tool_name="memory_get"):
                mid = self._extract_id_from_tool_call_pair(t, conv_history)
                if mid:
                    already[mid] = bool(getattr(t, "meta", {}).get("truncated", False))

            for t in conv_history.iter_tool_results(tool_name="memory_search"):
                for mid in self._extract_ids_from_search_result(t):
                    already[mid] = bool(getattr(t, "meta", {}).get("truncated", False))

        for entry in toc:
            if entry.id in already:
                entry.already_in_context = True
                entry.truncated_in_context = already[entry.id]

        return toc

    def _extract_id_from_tool_call_pair(
        self, result_turn, conv_history,
    ) -> str | None:
        meta = getattr(result_turn, "meta", {}) or {}
        call_id = meta.get("tool_call_id")
        if not call_id:
            return None
        window = getattr(conv_history, "window", [])
        for t in reversed(window):
            t_meta = getattr(t, "meta", {}) or {}
            if t_meta.get("trace_block") == "tool_call" \
                    and t_meta.get("tool_call_id") == call_id:
                args = t_meta.get("tool_args")
                if isinstance(args, dict) and "id" in args:
                    return str(args["id"])
                content = getattr(t, "content", "")
                m = re.search(
                    r'<argument\s+name=["\']id["\']\s*>([^<]+)</argument>',
                    content,
                )
                if m:
                    return m.group(1).strip()
                m = re.search(r'\bid["\']?\s*[:=]\s*["\']?(\w{6,})', content)
                if m:
                    return m.group(1)
                return None
        return None

    def _extract_ids_from_search_result(self, result_turn) -> list[str]:
        content = getattr(result_turn, "content", "")
        return re.findall(r'\bm_[a-f0-9]{8,16}\b', content)

    def render_toc_block(
        self,
        toc: list[TocEntry],
        injected_ids: "set[str] | None" = None,
    ) -> str:
        """Render a TOC list as an XML block for system-prompt injection."""
        if not toc:
            return ""
        project_attr = f' project="{self._active_project}"' if self._active_project else ""
        lines = [f'<memory_toc count="{len(toc)}"{project_attr}>']
        for e in toc:
            flag = ""
            if (injected_ids and e.id in injected_ids) or e.already_in_context:
                if e.truncated_in_context:
                    flag = " [already in context (truncated)]"
                else:
                    flag = " [injected into worker]" if (injected_ids and e.id in injected_ids) \
                        else " [already in context]"
            lines.append(f"  {e.id}{flag}: {e.retrieval_key}")
        lines.append("</memory_toc>")
        return "\n".join(lines)

    async def close(self) -> None:
        if self._store:
            self._store.close()
            self._store = None
