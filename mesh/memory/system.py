"""
Memory system orchestrator — two-tier architecture.

Maintains two layers:
1. Pool (P) — all memories ever created, persisted in SQLite. Unbounded.
2. Active set (S ⊂ P) — the top active_size entries selected by
   submodular optimization. Used for the router's <memory> block.

Worker retrieval (render_block_for_query) searches ALL of P by retrieval
key similarity (LLM-generated task descriptors), not just S. This means
workers can recall memories that the router doesn't show.

Public API:
- initialize() / close()
- reflect_on_completion(trigger, result, worker_id)
- render_block() — active set summaries for router prompt
- render_block_for_query(query, k) — top-K from FULL POOL for worker
- remember(id, full) — retrieve deeper tiers of any memory
- add_entry(...) — manual seeding
- delete_entry(id) — remove from pool
- get_entry(id) / list_entries()
"""

import asyncio
import dataclasses
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .embeddings import EmbeddingClient
from .selection import (
    compute_withholding_costs,
    cosine_sim,
    select_active_set,
    try_swap,
)
from .store import MemoryEntry, MemoryStore


# ── Memory Profile ─────────────────────────────────────────────

@dataclass(frozen=True)
class MemoryProfile:
    """Configuration for how memory is rendered into a prompt.

    Defines total token budget, slice allocations (Representative / Recent /
    Relevant), depth control, and similarity floor.
    """
    budget_tokens: int = 30_000

    # Slice allocation (should sum to 1.0)
    representative_pct: float = 0.40
    recent_pct: float = 0.30
    relevant_pct: float = 0.30

    # Depth control: how many entries per slice get full reflections
    representative_full_reflections: int = 0
    recent_full_reflections: int = 3
    relevant_full_reflections: int = 0
    relevant_top_traces: int = 0

    # Similarity floor for the Relevant slice
    similarity_floor: float = 0.25


LIGHT_PROFILE = MemoryProfile(
    budget_tokens=30_000,
    representative_pct=0.40,
    recent_pct=0.30,
    relevant_pct=0.30,
    representative_full_reflections=0,
    recent_full_reflections=3,
    relevant_full_reflections=0,
    relevant_top_traces=0,
    similarity_floor=0.25,
)

DEEP_PROFILE = MemoryProfile(
    budget_tokens=30_000,
    representative_pct=0.15,
    recent_pct=0.25,
    relevant_pct=0.60,
    representative_full_reflections=0,
    recent_full_reflections=2,
    relevant_full_reflections=5,
    relevant_top_traces=1,
    similarity_floor=0.25,
)

# Classifier profile: minimal context for routing decisions only.
# Hardcoded — not configurable via mesh.yaml.
CLASSIFIER_PROFILE = MemoryProfile(
    budget_tokens=2_000,
    representative_pct=0.0,
    recent_pct=1.0,
    relevant_pct=0.0,
    representative_full_reflections=0,
    recent_full_reflections=0,
    relevant_full_reflections=0,
    relevant_top_traces=0,
    similarity_floor=0.30,
)

# Backward compat aliases
ROUTER_PROFILE = LIGHT_PROFILE
WORKER_PROFILE = DEEP_PROFILE


def _build_profile(defaults: MemoryProfile, overrides) -> MemoryProfile:
    """Merge config overrides onto a default profile.

    Args:
        defaults: The built-in default profile (LIGHT_PROFILE or DEEP_PROFILE).
        overrides: A MemoryProfileConfig (from mesh.yaml) or None.

    Returns a new MemoryProfile with non-None overrides applied.
    """
    if overrides is None:
        return defaults
    fields = {}
    for f in dataclasses.fields(overrides):
        v = getattr(overrides, f.name)
        if v is not None:
            fields[f.name] = v
    return dataclasses.replace(defaults, **fields)


# XML overhead estimate: <entry> open tag + attributes + close tag
_XML_OVERHEAD_TOKENS = 30


def _entry_tokens(entry: MemoryEntry, depth: str) -> int:
    """Estimate tokens for rendering an entry at a given depth."""
    from mesh.llm import estimate_tokens

    if depth == "trace":
        text_tokens = estimate_tokens(entry.reflection or "") + estimate_tokens(entry.trace or "")
    elif depth == "full":
        text_tokens = estimate_tokens(entry.reflection or "")
    else:  # summary
        text_tokens = estimate_tokens(entry.summary or "")
    return _XML_OVERHEAD_TOKENS + text_tokens


@dataclass
class EpisodeStats:
    """Stats for a single router episode, used to decide whether to reflect."""
    tool_calls: int = 0
    num_user_visible_turns: int = 0
    total_user_visible_chars: int = 0
    agent_response_chars: int = 0
    has_errors: bool = False

    def merge(self, other: "EpisodeStats") -> None:
        """Accumulate another completion's stats into this session."""
        self.tool_calls += other.tool_calls
        self.num_user_visible_turns += other.num_user_visible_turns
        self.total_user_visible_chars += other.total_user_visible_chars
        self.agent_response_chars += other.agent_response_chars
        self.has_errors = self.has_errors or other.has_errors

logger = logging.getLogger(__name__)

# Reflection prompt template
REFLECTION_PROMPT = """You just completed a task. Reflect deeply on what happened.

<task_trigger>
{trigger}
</task_trigger>

<trace>
{trace}
</trace>

<outcome>
{outcome}
</outcome>

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
</retrieval_key>"""


class MemorySystem:
    """
    Two-tier episodic memory system for mesh agents.

    Pool (P): all memories, stored in SQLite. Never evicted except by
    explicit delete.

    Active set (S): subset of P selected by submodular optimization.
    Size bounded by active_size. Used for router prompt rendering.

    Worker retrieval searches all of P by retrieval key similarity.
    """

    def __init__(
        self,
        nickname: str,
        llm_client,
        active_size: int = 30,
        pool_max_entries: int = 1000,
        embedding_backend: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        reflection_min_tools: int = 3,
        retrieval_k: int = 5,
        worker_full_reflections: int = 2,
        router_full_reflections: int = 0,
        router_recent_reflections: int = 3,
        worker_recent_reflections: int = 2,
        trace_max_tokens: int = 2000,
        reflection_max_tokens: int = 500,
        # Extended reflection criteria
        reflection_min_discussion_turns: int = 4,
        reflection_min_discussion_chars: int = 1500,
        reflection_min_brainstorm_response_chars: int = 1500,
        reflection_max_brainstorm_tools: int = 2,
        reflection_cooldown_secs: int = 300,
        # Profile configs (optional — None means use built-in defaults)
        light_profile_config=None,
        deep_profile_config=None,
        # Backward compat aliases
        router_profile_config=None,
        worker_profile_config=None,
    ):
        self._nickname = nickname
        self._llm_client = llm_client
        self._active_size = active_size
        self._pool_max_entries = pool_max_entries
        self._reflection_min_tools = reflection_min_tools
        self._retrieval_k = retrieval_k
        self._worker_full_reflections = worker_full_reflections
        self._router_full_reflections = router_full_reflections
        self._router_recent_reflections = router_recent_reflections
        self._worker_recent_reflections = worker_recent_reflections
        self._trace_max_tokens = trace_max_tokens
        self._reflection_max_tokens = reflection_max_tokens

        # Extended reflection criteria
        self._reflection_min_discussion_turns = reflection_min_discussion_turns
        self._reflection_min_discussion_chars = reflection_min_discussion_chars
        self._reflection_min_brainstorm_response_chars = reflection_min_brainstorm_response_chars
        self._reflection_max_brainstorm_tools = reflection_max_brainstorm_tools
        self._reflection_cooldown_secs = reflection_cooldown_secs
        self._last_reflection_time: float = 0.0  # monotonic timestamp

        self._store: MemoryStore | None = None
        self._embedder = EmbeddingClient(
            backend=embedding_backend,
            model=embedding_model,
        )

        # Build rendering profiles (new names take priority over old)
        _light_cfg = light_profile_config or router_profile_config
        _deep_cfg = deep_profile_config or worker_profile_config
        self.light_profile = _build_profile(LIGHT_PROFILE, _light_cfg)
        self.deep_profile = _build_profile(DEEP_PROFILE, _deep_cfg)
        # Backward compat aliases
        self.router_profile = self.light_profile
        self.worker_profile = self.deep_profile

        # Query embedding cache (keyed by text hash, avoids repeated API calls)
        self._query_embedding_cache: dict[int, np.ndarray] = {}

        # Personality cache (loaded from SQLite on init, avoids cross-thread access)
        self._personality_cache: str = ""

        # Two-tier in-memory state
        self._pool: list[MemoryEntry] = []      # All entries (P)
        self._active_ids: set[str] = set()       # IDs of entries in active set (S)
        self._active_weights: dict[str, float] = {}  # Withholding costs for S
        self._active_f_s: float = 0.0            # f(S) for active set

    @property
    def classifier_profile(self) -> MemoryProfile:
        """Minimal profile for router classification — not configurable."""
        return CLASSIFIER_PROFILE

    @property
    def active_entries(self) -> list[MemoryEntry]:
        """Entries in the active set S, preserving pool order."""
        return [e for e in self._pool if e.id in self._active_ids]

    async def initialize(self) -> None:
        """Open the store, load all entries into pool, compute active set."""
        self._store = MemoryStore(self._nickname)
        self._pool = self._store.load()
        self._personality_cache = self._store.get_personality()
        self._reselect_active_set()
        self._prune_pool()
        logger.info(
            f"Memory system initialized for '{self._nickname}': "
            f"{len(self._pool)} entries in pool, "
            f"{len(self._active_ids)} in active set"
        )

    def get_diagnostics(self) -> dict:
        """Return structured diagnostic data for status reporting."""
        import time as _time
        return {
            "pool_size": len(self._pool),
            "active_set_size": len(self._active_ids),
            "active_set_target": self._active_size,
            "pool_max_entries": self._pool_max_entries,
            "last_reflection_ago_seconds": (
                round(_time.monotonic() - self._last_reflection_time, 1)
                if self._last_reflection_time > 0 else None
            ),
            "reflection_cooldown_seconds": self._reflection_cooldown_secs,
            "retrieval_k": self._retrieval_k,
        }

    # ── Personality ──────────────────────────────────────────────

    def get_personality(self) -> str:
        """Get the agent's personality text (from in-memory cache)."""
        return self._personality_cache

    def set_personality(self, text: str) -> None:
        """Set the agent's personality text in the store and cache.

        NOTE: This writes to SQLite — call from the event loop thread
        (i.e. from an async tool handler) to avoid cross-thread errors.
        """
        self._personality_cache = text
        if self._store is not None:
            self._store.set_personality(text)

    def seed_personality(self, config_text: str) -> None:
        """Seed personality from config if the DB has no personality yet.

        Only writes if the DB personality is empty AND config_text is non-empty.
        Once the agent sets its own personality, the config seed is ignored.

        Must be called from the event loop thread (during init).
        """
        if not config_text or self._store is None:
            return
        if not self._personality_cache:
            self._store.set_personality(config_text)
            self._personality_cache = config_text
            logger.info(f"Personality seeded from config for '{self._nickname}'")

    async def close(self) -> None:
        """Close the store."""
        if self._store:
            self._store.close()
            self._store = None

    # ── Active set management ──────────────────────────────────

    def _reselect_active_set(self) -> None:
        """Run full greedy submodular selection over the pool."""
        if not self._pool:
            self._active_ids = set()
            self._active_weights = {}
            self._active_f_s = 0.0
            return

        embs = [e.reflection_embedding for e in self._pool
                if e.reflection_embedding is not None]

        # Map from embedding index back to pool entry
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

        # Convert embedding indices to entry IDs
        self._active_ids = set()
        self._active_weights = {}
        for sel_pos, emb_idx in enumerate(selected_indices):
            pool_idx = emb_to_pool[emb_idx]
            entry = self._pool[pool_idx]
            self._active_ids.add(entry.id)
            self._active_weights[entry.id] = weights[sel_pos]
            entry.weight = weights[sel_pos]
        self._active_f_s = f_s

        # Zero out weights for pool-only entries
        for e in self._pool:
            if e.id not in self._active_ids:
                e.weight = 0.0

        # Persist weights
        if self._store:
            self._store.update_weights_batch(
                {e.id: e.weight for e in self._pool}
            )

    def _incremental_active_update(self, new_entry: MemoryEntry) -> bool:
        """
        Incrementally update the active set when a new entry is added to the pool.

        Uses try_swap against the current active set S.
        Returns True if the new entry entered the active set.
        """
        active = self.active_entries
        # Filter consistently so embs and weights stay aligned
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
                # Remove evicted entry from active set (NOT from pool)
                evicted = active_with_embs[evict_idx]
                self._active_ids.discard(evicted.id)
                del self._active_weights[evicted.id]
                evicted.weight = 0.0
                logger.info(
                    f"Memory active set swap: '{evicted.id}' out, "
                    f"'{new_entry.id}' in"
                )
            else:
                logger.info(
                    f"Memory accepted into active set: '{new_entry.id}' "
                    f"({len(self._active_ids) + 1}/{self._active_size})"
                )

            # Add new entry to active set
            self._active_ids.add(new_entry.id)

            # Rebuild active weights from new_weights
            updated_active = self.active_entries
            self._active_weights = {}
            for i, e in enumerate(updated_active):
                e.weight = new_weights[i]
                self._active_weights[e.id] = new_weights[i]
            self._active_f_s = new_f_s

            # Persist weights
            if self._store:
                self._store.update_weights_batch(
                    {e.id: e.weight for e in self._pool}
                )

            return True
        else:
            logger.info(f"Memory candidate '{new_entry.id}' stays pool-only")
            return False

    def _prune_pool(self) -> int:
        """
        Prune oldest pool-only entries when pool exceeds pool_max_entries.

        Active entries (in S) are never pruned. Only pool-only entries are
        candidates, sorted by created_at ascending (oldest first).

        Returns the number of entries pruned.
        """
        if len(self._pool) <= self._pool_max_entries:
            return 0

        excess = len(self._pool) - self._pool_max_entries

        # Collect pool-only entries sorted by age (oldest first)
        pool_only = sorted(
            [e for e in self._pool if e.id not in self._active_ids],
            key=lambda e: e.created_at,
        )

        to_remove = pool_only[:excess]
        if not to_remove:
            # All entries are active — can't prune
            return 0

        remove_ids = {e.id for e in to_remove}
        self._pool = [e for e in self._pool if e.id not in remove_ids]
        for entry_id in remove_ids:
            self._store.delete(entry_id)

        logger.info(
            f"Memory pool pruned: removed {len(remove_ids)} oldest pool-only entries "
            f"(pool size: {len(self._pool)})"
        )
        return len(remove_ids)

    # ── Reflection ──────────────────────────────────────────────

    def should_reflect(self, result, stats: EpisodeStats | None = None) -> bool:
        """
        Determine if a worker result warrants memory reflection.

        Three-path significance filter (any one triggers reflection):
        1. Tool-heavy: tool_calls >= min_tools
        2. Extended discussion: num_user_visible_turns >= 4 AND total_user_visible_chars >= 1500
        3. Brainstorm: tool_calls <= 2 AND agent_response_chars >= 1500 AND num_user_visible_turns >= 2

        Plus: cooldown guard to prevent reflection storms.

        If stats is None, falls back to legacy tool-counting from result.context.
        """
        # Cooldown guard
        now = time.monotonic()
        if (now - self._last_reflection_time) < self._reflection_cooldown_secs:
            logger.debug("Memory reflection skipped: cooldown active")
            return False

        if stats is not None:
            return self._should_reflect_from_stats(stats)

        # Legacy fallback: count tools from result context
        return self._should_reflect_legacy(result)

    def _should_reflect_from_stats(self, stats: EpisodeStats) -> bool:
        """Three-path significance check using pre-computed EpisodeStats."""
        # Path 1: Tool-heavy
        if stats.tool_calls >= self._reflection_min_tools:
            return True

        # Path 2: Extended discussion
        if (
            stats.num_user_visible_turns >= self._reflection_min_discussion_turns
            and stats.total_user_visible_chars >= self._reflection_min_discussion_chars
        ):
            return True

        # Path 3: Brainstorm (few/no tools, long agent response, multi-turn)
        if (
            stats.tool_calls <= self._reflection_max_brainstorm_tools
            and stats.agent_response_chars >= self._reflection_min_brainstorm_response_chars
            and stats.num_user_visible_turns >= 2
        ):
            return True

        # Also reflect on errors
        if stats.has_errors:
            return True

        return False

    def _should_reflect_legacy(self, result) -> bool:
        """Legacy fallback: count tools from result.context."""
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
        self,
        trigger: str,
        result,
        worker_id: str,
        topic_label: str = "",
    ) -> None:
        """
        Reflect on a completed task and store the memory in the pool.

        The entry always goes into the pool. The incremental active set
        update decides whether it also enters S.

        Runs asynchronously — should be called via asyncio.create_task().
        """
        try:
            # Record reflection time for cooldown
            self._last_reflection_time = time.monotonic()

            # Step 1: Construct clipped trace
            trace = self._build_trace(result)

            # Step 2: LLM reflection call
            outcome_text = result.response or "(no response)"
            prompt = REFLECTION_PROMPT.format(
                trigger=trigger,
                trace=trace,
                outcome=outcome_text[:2000],
            )
            llm_response = await asyncio.wait_for(
                self._llm_client.complete(prompt),
                timeout=180,
            )

            # Step 3: Parse reflection output
            reflection = _extract_tag(llm_response, "reflection")
            summary = _extract_tag(llm_response, "summary")
            tags_str = _extract_tag(llm_response, "tags")
            outcome_label = _extract_tag(llm_response, "outcome_label")
            retrieval_key = _extract_tag(llm_response, "retrieval_key")

            if not reflection or not summary:
                logger.warning("Memory reflection produced empty output, skipping")
                return

            # Fallback: use summary as retrieval key if LLM didn't produce one
            if not retrieval_key:
                retrieval_key = summary

            tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
            outcome = outcome_label.strip() if outcome_label else "success"
            if outcome not in ("success", "partial", "failure"):
                outcome = "success"

            # Step 4: Compute embeddings (reflection + retrieval key)
            emb_results = await asyncio.wait_for(
                self._embedder.embed_batch_to_arrays([reflection, retrieval_key]),
                timeout=30,
            )
            reflection_emb = emb_results[0]
            retrieval_key_emb = emb_results[1]

            # Step 5: Build entry
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
                reflection_embedding=reflection_emb,
                retrieval_key_embedding=retrieval_key_emb,
                weight=0.0,
            )

            # Step 6: Always store in pool
            self._pool.append(entry)
            self._store.insert(entry)

            # Step 7: Incremental active set update
            self._incremental_active_update(entry)

            # Step 8: Prune pool if over cap
            self._prune_pool()

        except asyncio.TimeoutError:
            logger.warning("Memory reflection timed out (120s)")
        except Exception:
            logger.error("Memory reflection failed", exc_info=True)

    def _build_trace(self, result) -> str:
        """
        Construct a clipped trace from worker result context.

        Extracts tool calls and their results, truncating each to a
        reasonable size. Total trace capped at ~trace_max_tokens chars.
        """
        lines = []
        char_budget = self._trace_max_tokens * 4  # rough chars-to-tokens

        for entry in getattr(result, "context", []):
            msg = entry.message if hasattr(entry, "message") else entry
            metadata = getattr(msg, "metadata", {}) or {}
            content = getattr(msg, "content", "")

            if metadata.get("tool_calls"):
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
                result_str = str(content)[:500]
                lines.append(f"[RESULT] {result_str}")

        trace = "\n".join(lines)
        if len(trace) > char_budget:
            trace = trace[:char_budget] + "\n... (trace truncated)"
        return trace

    # ── Rendering ───────────────────────────────────────────────

    async def render_block(self, query: str | None = None) -> str:
        """Legacy: render for router using default light profile.

        Breaking change: was sync, now async. All callers must add await.
        """
        return await self.render(self.light_profile, query=query)

    def _get_recent_entries(
        self, n: int, exclude_ids: set[str] | None = None
    ) -> list[MemoryEntry]:
        """
        Return the N most recent entries from the pool, excluding given IDs.

        Sorted by created_at descending (newest first).
        """
        if n <= 0 or not self._pool:
            return []
        exclude = exclude_ids or set()
        candidates = [e for e in self._pool if e.id not in exclude]
        candidates.sort(key=lambda e: e.created_at, reverse=True)
        return candidates[:n]

    async def render_block_for_query(
        self,
        query: str,
        k: int | None = None,
        n_full_reflections: int | None = None,
        tag: str | None = None,
    ) -> str:
        """Legacy: render for worker using default deep profile.

        Note: k and n_full_reflections are accepted but silently ignored —
        the deep profile's token budget and depth rules replace count-based
        params. Callers continue to work, just with budget-based selection.
        """
        return await self.render(self.deep_profile, query=query, tag=tag)

    # ── Profile-based rendering ──────────────────────────────────

    async def render(
        self, profile: MemoryProfile, query: str | None = None, tag: str | None = None,
    ) -> str:
        """
        Render memory entries as XML block using the given profile.

        Three-slice rendering with token budgets:
        1. Representative — from active set S, sorted by weight desc
        2. Recent — from pool, sorted by created_at desc
        3. Relevant — from pool, sorted by cosine similarity to query

        Cross-slice dedup ensures each entry appears at most once.
        Depth escalation gives entries the max depth across all qualifying slices.
        Unused Relevant budget redistributes to Recent, then Representative.

        Args:
            profile: Specifies total budget, slice allocations, depth controls.
            query: Optional query for the Relevant slice. If None, Relevant
                   is skipped and its budget redistributes.
            tag: Optional tag filter. If set, only entries containing this
                 exact tag are included in all slices.

        Returns XML string or empty string if no entries.
        """
        pool = self._pool
        if tag:
            pool = [e for e in pool if tag in e.tags]
        if not pool:
            return ""
        if profile.budget_tokens <= 0:
            return ""

        # Step 1: Compute per-slice budgets
        repr_budget = int(profile.budget_tokens * profile.representative_pct)
        recent_budget = int(profile.budget_tokens * profile.recent_pct)
        relevant_budget = int(profile.budget_tokens * profile.relevant_pct)

        # Step 2: Fill Representative (active set by weight desc)
        pool_ids = {e.id for e in pool} if tag else None
        active_src = [e for e in self.active_entries if e.id in pool_ids] if pool_ids is not None else self.active_entries
        active_sorted = sorted(
            active_src,
            key=lambda e: self._active_weights.get(e.id, 0.0),
            reverse=True,
        )
        repr_entries: list[tuple[MemoryEntry, str]] = []  # (entry, depth)
        repr_used = 0
        seen_ids: set[str] = set()
        for idx, entry in enumerate(active_sorted):
            depth = "full" if idx < profile.representative_full_reflections and entry.reflection else "summary"
            cost = _entry_tokens(entry, depth)
            if repr_used + cost > repr_budget:
                break
            repr_entries.append((entry, depth))
            repr_used += cost
            seen_ids.add(entry.id)

        # Step 3: Fill Recent (pool by created_at desc, excluding seen)
        recent_candidates = sorted(
            [e for e in pool if e.id not in seen_ids],
            key=lambda e: e.created_at,
            reverse=True,
        )
        recent_entries: list[tuple[MemoryEntry, str]] = []
        recent_used = 0
        for idx, entry in enumerate(recent_candidates):
            depth = "full" if idx < profile.recent_full_reflections and entry.reflection else "summary"
            cost = _entry_tokens(entry, depth)
            if recent_used + cost > recent_budget:
                break
            recent_entries.append((entry, depth))
            recent_used += cost
            seen_ids.add(entry.id)

        # Step 4: Fill Relevant (cosine similarity, excluding seen)
        relevant_entries: list[tuple[MemoryEntry, str, float]] = []  # (entry, depth, sim)
        relevant_used = 0
        if query is not None:
            query_emb = await self._get_query_embedding(query)
            if query_emb is not None:
                scored = []
                for entry in pool:
                    if entry.id in seen_ids:
                        continue
                    if entry.retrieval_key_embedding is None:
                        continue
                    sim = cosine_sim(query_emb, entry.retrieval_key_embedding)
                    if sim >= profile.similarity_floor:
                        scored.append((sim, entry))
                scored.sort(key=lambda x: x[0], reverse=True)

                rel_idx = 0
                for sim, entry in scored:
                    if rel_idx < profile.relevant_top_traces and entry.trace and entry.reflection:
                        depth = "trace"
                    elif rel_idx < profile.relevant_full_reflections and entry.reflection:
                        depth = "full"
                    else:
                        depth = "summary"
                    cost = _entry_tokens(entry, depth)
                    if relevant_used + cost > relevant_budget:
                        break
                    relevant_entries.append((entry, depth, sim))
                    relevant_used += cost
                    seen_ids.add(entry.id)
                    rel_idx += 1

        # Step 5: Depth escalation
        # Entries claimed by an earlier slice may qualify for deeper depth
        # in a later slice. We escalate to the max depth across all qualifying
        # slices. Depth ordering: summary < full < trace.
        depth_rank = {"summary": 0, "full": 1, "trace": 2}
        rank_to_depth = {0: "summary", 1: "full", 2: "trace"}

        # Build set of entry IDs that would qualify for Recent full-reflection
        recent_by_time = sorted(pool, key=lambda e: e.created_at, reverse=True)
        recent_full_ids = {
            e.id for e in recent_by_time[:profile.recent_full_reflections]
            if e.reflection
        }

        # Build similarity scores for ALL pool entries if we have a query,
        # so we can check Relevant qualification for deduped entries too.
        relevant_full_ids: set[str] = set()
        relevant_trace_ids: set[str] = set()
        if query is not None:
            query_emb = await self._get_query_embedding(query)
            if query_emb is not None:
                all_scored = []
                for entry in pool:
                    if entry.retrieval_key_embedding is None:
                        continue
                    sim = cosine_sim(query_emb, entry.retrieval_key_embedding)
                    if sim >= profile.similarity_floor:
                        all_scored.append((sim, entry))
                all_scored.sort(key=lambda x: x[0], reverse=True)
                for rank, (_, entry) in enumerate(all_scored):
                    if rank < profile.relevant_top_traces and entry.trace and entry.reflection:
                        relevant_trace_ids.add(entry.id)
                    if rank < profile.relevant_full_reflections and entry.reflection:
                        relevant_full_ids.add(entry.id)

        def _max_depth(entry: MemoryEntry, current: str) -> str:
            """Return the maximum depth this entry qualifies for."""
            best = depth_rank[current]
            if entry.id in recent_full_ids:
                best = max(best, depth_rank["full"])
            if entry.id in relevant_full_ids:
                best = max(best, depth_rank["full"])
            if entry.id in relevant_trace_ids:
                best = max(best, depth_rank["trace"])
            return rank_to_depth[best]

        # Escalate Representative entries
        repr_escalation_cost = 0
        for i, (entry, current_depth) in enumerate(repr_entries):
            new_depth = _max_depth(entry, current_depth)
            if new_depth != current_depth:
                old_cost = _entry_tokens(entry, current_depth)
                new_cost = _entry_tokens(entry, new_depth)
                repr_escalation_cost += (new_cost - old_cost)
                repr_entries[i] = (entry, new_depth)
        repr_used += repr_escalation_cost

        # Escalate Recent entries
        recent_escalation_cost = 0
        for i, (entry, current_depth) in enumerate(recent_entries):
            new_depth = _max_depth(entry, current_depth)
            if new_depth != current_depth:
                old_cost = _entry_tokens(entry, current_depth)
                new_cost = _entry_tokens(entry, new_depth)
                recent_escalation_cost += (new_cost - old_cost)
                recent_entries[i] = (entry, new_depth)
        recent_used += recent_escalation_cost

        # Step 6: Redistribute unused Relevant budget
        leftover = relevant_budget - relevant_used
        if leftover > 0:
            # Extend Recent
            extra_recent_candidates = sorted(
                [e for e in pool if e.id not in seen_ids],
                key=lambda e: e.created_at,
                reverse=True,
            )
            for entry in extra_recent_candidates:
                if leftover <= 0:
                    break
                depth = "summary"
                cost = _entry_tokens(entry, depth)
                if cost <= leftover:
                    recent_entries.append((entry, depth))
                    recent_used += cost
                    leftover -= cost
                    seen_ids.add(entry.id)

        if leftover > 0:
            # Extend Representative
            extra_repr_candidates = sorted(
                [e for e in pool
                 if e.id not in seen_ids and e.reflection_embedding is not None],
                key=lambda e: self._active_weights.get(e.id, 0.0),
                reverse=True,
            )
            for entry in extra_repr_candidates:
                if leftover <= 0:
                    break
                depth = "summary"
                cost = _entry_tokens(entry, depth)
                if cost <= leftover:
                    repr_entries.append((entry, depth))
                    repr_used += cost
                    leftover -= cost
                    seen_ids.add(entry.id)

        # Step 7: Emit XML
        if not repr_entries and not recent_entries and not relevant_entries:
            return ""

        parts = ["<memory>"]

        for entry, depth in repr_entries:
            parts.append(self._render_entry_xml(entry, depth, "representative"))

        for entry, depth in recent_entries:
            parts.append(self._render_entry_xml(entry, depth, "recent"))

        for entry, depth, sim in relevant_entries:
            parts.append(self._render_entry_xml(entry, depth, "relevant", similarity=sim))

        parts.append("</memory>")
        return "\n".join(parts)

    def _render_entry_xml(
        self,
        entry: MemoryEntry,
        depth: str,
        source: str,
        similarity: float | None = None,
    ) -> str:
        """Render a single memory entry as XML."""
        tags_str = ", ".join(entry.tags) if entry.tags else ""
        date_str = entry.created_at.strftime("%Y-%m-%d")
        timestamp_str = entry.created_at.strftime("%Y-%m-%d %H:%M")

        sim_attr = f' similarity="{similarity:.2f}"' if similarity is not None else ""
        header = (
            f'<entry id="{entry.id}" date="{date_str}" '
            f'tags="{tags_str}" outcome="{entry.outcome}" '
            f'depth="{depth}" source="{source}"{sim_attr}>'
        )

        lines = [header]
        if depth in ("full", "trace") and entry.reflection:
            lines.append(f"[{entry.id} | {timestamp_str}]")
            lines.append(entry.reflection)
            if depth == "trace" and entry.trace:
                lines.append(entry.trace)
        else:
            lines.append(entry.summary)
        lines.append("</entry>")
        return "\n".join(lines)

    async def _get_query_embedding(self, query: str) -> np.ndarray | None:
        """Get embedding for a query, using cache to avoid repeated API calls."""
        cache_key = hash(query)
        if cache_key in self._query_embedding_cache:
            return self._query_embedding_cache[cache_key]
        try:
            emb = await self._embedder.embed_to_array(query)
            self._query_embedding_cache[cache_key] = emb
            # Keep cache small — only cache the last few queries
            if len(self._query_embedding_cache) > 10:
                oldest = next(iter(self._query_embedding_cache))
                del self._query_embedding_cache[oldest]
            return emb
        except Exception:
            logger.error("Failed to embed query for memory retrieval", exc_info=True)
            return None

    # ── Retrieval ───────────────────────────────────────────────

    def remember(self, entry_id: str, full: bool = False) -> str | None:
        """
        Retrieve deeper tiers of a memory entry (searches full pool).

        full=False: returns reflection only (~500 tokens)
        full=True:  returns reflection + trace (~2K tokens)
        """
        entry = self._find_entry(entry_id)
        if entry is None:
            return None

        if full:
            return (
                f"## Reflection\n\n{entry.reflection}\n\n"
                f"## Trace\n\n{entry.trace}"
            )
        return entry.reflection

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """Get a memory entry by ID (searches full pool)."""
        return self._find_entry(entry_id)

    def list_entries(self) -> list[MemoryEntry]:
        """List all memory entries in the pool."""
        return list(self._pool)

    def is_active(self, entry_id: str) -> bool:
        """Check if an entry is in the active set."""
        return entry_id in self._active_ids

    async def add_entry(
        self,
        summary: str,
        reflection: str = "",
        trace: str = "",
        tags: list[str] | None = None,
        outcome: str = "success",
    ) -> tuple[MemoryEntry, bool]:
        """
        Manually add a memory entry (for seeding via tools).

        The entry always goes into the pool. Returns (entry, in_active_set).
        """
        # Embed reflection and summary (use summary as retrieval key for manual entries)
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
            trigger=summary,  # Use summary as trigger for manual entries
            retrieval_key=summary,  # Use summary as retrieval key for manual entries
            tags=tags or [],
            outcome=outcome,
            reflection_embedding=embs[0],
            retrieval_key_embedding=embs[1],
            weight=0.0,
        )

        # Always store in pool + SQLite
        self._pool.append(entry)
        self._store.insert(entry)

        # Incremental active set update
        in_active = self._incremental_active_update(entry)

        # Prune pool if over cap
        self._prune_pool()

        return entry, in_active

    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a memory entry from the pool and store.

        If the deleted entry was in the active set, triggers a full
        reselection to fill the gap.
        """
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
            # Full reselection to fill the gap
            self._active_ids.discard(entry_id)
            if entry_id in self._active_weights:
                del self._active_weights[entry_id]
            self._reselect_active_set()
        # If not active, pool just shrinks — no reselection needed

        return True

    def _find_entry(self, entry_id: str) -> MemoryEntry | None:
        """Find entry in the full pool."""
        for e in self._pool:
            if e.id == entry_id:
                return e
        return None


def _extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag> and </tag>."""
    import re
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
