#!/usr/bin/env python3
"""
Memory pool backfill — populate memory pool from agent conversation history.

Parses the agent's history JSON into episodes, applies significance filters
(tool-heavy and/or extended-discussion), and optionally generates LLM
reflections and stores them in the memory pool.

Usage:
  # Dry-run: see candidate episodes without LLM calls or DB writes
  python -m mesh.scripts.backfill_memory --agent coder --nickname claude-sobek --dry-run

  # Live: generate reflections and store in memory pool
  python -m mesh.scripts.backfill_memory --agent coder --nickname claude-sobek

  # With filters
  python -m mesh.scripts.backfill_memory --agent coder --nickname claude-sobek \
      --min-tools 5 --since 2026-01-26 --max-episodes 50
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mesh.memory.store import MemoryEntry, MemoryStore
from mesh.memory.embeddings import EmbeddingClient
from mesh.memory.selection import cosine_sim, select_active_set
from mesh.memory.system import REFLECTION_PROMPT, _extract_tag
from mesh.llm import LLMClient, LLMConfig

logger = logging.getLogger(__name__)

# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class Episode:
    """A single episode extracted from agent conversation history."""
    index: int
    timestamp: str  # ISO timestamp of the trigger message
    trigger: str  # Content of the first user/channel message
    tool_count: int  # Number of tool calls (CC + mesh)
    tool_summary: str  # e.g. "Read x4, Bash x3"
    cc_tool_blocks: list[str] = field(default_factory=list)
    mesh_tool_lines: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)
    num_turns: int = 0  # User-visible turns only (user↔agent messages)
    total_chars: int = 0  # Sum of user-visible message content lengths
    agent_response_chars: int = 0  # Sum of agent response content lengths
    significance_reason: str = ""  # "tool-heavy" | "extended-discussion" | "brainstorm" | "both"


@dataclass
class ReflectionResult:
    """Parsed output from a reflection LLM call."""
    summary: str
    reflection: str
    tags: list[str]
    outcome_label: str
    retrieval_key: str = ""


# ── History loading ────────────────────────────────────────────────────

def load_history(nickname: str) -> list[dict]:
    """Load agent history JSON from ~/.mesh/history/agent-<nickname>.json."""
    from mesh.paths import resolve_path
    path = resolve_path(f"~/.mesh/history/agent-{nickname}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"History file not found: {path}")
    with open(path) as f:
        return json.load(f)


# ── Episode parsing ───────────────────────────────────────────────────

def _is_user_trigger(entry: dict) -> bool:
    """Check if an entry is an incoming user/channel message (episode start)."""
    if entry.get("direction") != "incoming":
        return False
    msg = entry.get("message", {})
    from_node = msg.get("from_node", "")
    to_node = msg.get("to_node", "")
    # User-initiated messages
    if from_node.startswith("user:"):
        return True
    # Channel messages from users
    if to_node.startswith("channel:") and from_node.startswith("user:"):
        return True
    return False


def _get_content(entry: dict) -> str:
    """Extract message content as string."""
    msg = entry.get("message", {})
    content = msg.get("content", "")
    return content if isinstance(content, str) else str(content)


def _get_timestamp(entry: dict) -> str:
    """Extract timestamp from entry."""
    msg = entry.get("message", {})
    return msg.get("timestamp", "")


def parse_episodes(history: list[dict], session_gap_secs: int = 900) -> list[Episode]:
    """Group history entries into episodes using time-gap-based session merging.

    A new episode starts only when there's a gap of >= session_gap_secs between
    consecutive user messages. Follow-up messages within the gap are part of
    the same episode/session.
    """
    episodes: list[Episode] = []
    current_entries: list[dict] = []
    last_user_ts: datetime | None = None
    ep_index = 0

    for entry in history:
        if _is_user_trigger(entry):
            entry_ts = _parse_timestamp(_get_timestamp(entry))

            # Start a new episode if: no current episode, or gap exceeds threshold
            should_split = False
            if not current_entries:
                should_split = True
            elif entry_ts and last_user_ts:
                gap = (entry_ts - last_user_ts).total_seconds()
                should_split = gap >= session_gap_secs
            elif not entry_ts or not last_user_ts:
                # Can't determine gap — start new episode to be safe
                should_split = True

            if should_split and current_entries:
                # Finalize previous episode
                ep = _build_episode(current_entries, ep_index)
                if ep is not None:
                    episodes.append(ep)
                    ep_index += 1
                current_entries = [entry]
            elif should_split:
                current_entries = [entry]
            else:
                # Continue current episode (within session gap)
                current_entries.append(entry)

            last_user_ts = entry_ts
        else:
            if current_entries:
                current_entries.append(entry)
            # Entries before the first trigger are ignored

    # Finalize last episode
    if current_entries:
        ep = _build_episode(current_entries, ep_index)
        if ep is not None:
            episodes.append(ep)

    return episodes


def _parse_timestamp(ts: str) -> datetime | None:
    """Parse ISO timestamp string to datetime, or None on failure."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _build_episode(entries: list[dict], index: int) -> Episode | None:
    """Build an Episode from a group of history entries."""
    if not entries:
        return None

    # Trigger is the first entry
    trigger_content = _get_content(entries[0])
    timestamp = _get_timestamp(entries[0])

    cc_tool_blocks: list[str] = []
    mesh_tool_lines: list[str] = []
    responses: list[str] = []
    tool_type_counts: dict[str, int] = {}
    num_turns = 0
    total_chars = 0
    agent_response_chars = 0

    for entry in entries:
        content = _get_content(entry)
        direction = entry.get("direction", "")
        msg = entry.get("message", {})
        from_node = msg.get("from_node", "")
        to_node = msg.get("to_node", "")

        # CC Tool Activity blocks
        if content.startswith("[CC Tool Activity]"):
            cc_tool_blocks.append(content)
            # Count individual tool calls within CC blocks
            for match in re.finditer(r"\[cc:(\w+)\]", content):
                tool_name = match.group(1)
                tool_type_counts[tool_name] = tool_type_counts.get(tool_name, 0) + 1

        # Mesh tool calls
        elif direction == "outgoing" and to_node == "internal" and content.startswith("[Tool:"):
            mesh_tool_lines.append(content)
            # Extract tool names like [Tool: bash_exec(...)]
            for match in re.finditer(r"\[Tool:\s*(\w+)", content):
                tool_name = match.group(1)
                tool_type_counts[tool_name] = tool_type_counts.get(tool_name, 0) + 1

        # User-visible messages: user→agent or agent→user/channel
        elif from_node.startswith("user:"):
            num_turns += 1
            total_chars += len(content)

        elif direction == "outgoing" and (
            to_node.startswith("user:") or to_node.startswith("channel:")
        ):
            responses.append(content)
            num_turns += 1
            total_chars += len(content)
            agent_response_chars += len(content)

    tool_count = sum(tool_type_counts.values())

    # Build tool summary string
    if tool_type_counts:
        sorted_tools = sorted(tool_type_counts.items(), key=lambda x: -x[1])
        tool_summary = ", ".join(f"{name} x{count}" for name, count in sorted_tools)
    else:
        tool_summary = "none"

    return Episode(
        index=index,
        timestamp=timestamp,
        trigger=trigger_content,
        tool_count=tool_count,
        tool_summary=tool_summary,
        cc_tool_blocks=cc_tool_blocks,
        mesh_tool_lines=mesh_tool_lines,
        responses=responses,
        num_turns=num_turns,
        total_chars=total_chars,
        agent_response_chars=agent_response_chars,
    )


def filter_episodes(
    episodes: list[Episode],
    min_tools: int = 3,
    min_discussion_turns: int = 4,
    min_discussion_chars: int = 1500,
    max_brainstorm_tools: int = 2,
    min_brainstorm_response_chars: int = 1500,
    since: str | None = None,
) -> list[Episode]:
    """Apply significance filters. Returns qualifying episodes with reason set.

    Three paths to significance (aligned with online should_reflect):
    1. tool-heavy: tool_calls >= min_tools
    2. extended-discussion: num_turns >= min_discussion_turns AND total_chars >= min_discussion_chars
    3. brainstorm: tool_count <= max_brainstorm_tools AND agent_response_chars >= min_brainstorm_response_chars AND num_turns >= 2
    """
    filtered = []

    for ep in episodes:
        # Date filter
        if since and ep.timestamp:
            try:
                ep_dt = datetime.fromisoformat(ep.timestamp.replace("Z", "+00:00"))
                since_dt = datetime.fromisoformat(since + "T00:00:00+00:00")
                if ep_dt < since_dt:
                    continue
            except (ValueError, TypeError):
                pass  # If we can't parse, include it

        # Skip trivial triggers
        trigger_stripped = ep.trigger.strip()
        if len(trigger_stripped) < 10 and "?" not in trigger_stripped:
            continue

        # Check all three criteria (same logic as online should_reflect)
        is_tool_heavy = ep.tool_count >= min_tools
        is_discussion = (
            ep.num_turns >= min_discussion_turns
            and ep.total_chars >= min_discussion_chars
        )
        is_brainstorm = (
            ep.tool_count <= max_brainstorm_tools
            and ep.agent_response_chars >= min_brainstorm_response_chars
            and ep.num_turns >= 2
        )

        # Determine reason (prefer the most specific combo)
        reasons = []
        if is_tool_heavy:
            reasons.append("tool-heavy")
        if is_discussion:
            reasons.append("extended-discussion")
        if is_brainstorm and not is_discussion:
            # Only add brainstorm if not already captured by extended-discussion
            reasons.append("brainstorm")

        if not reasons:
            continue

        if len(reasons) > 1:
            ep.significance_reason = "both"
        else:
            ep.significance_reason = reasons[0]

        filtered.append(ep)

    return filtered


# ── Trace and content construction ─────────────────────────────────────

def build_trace(episode: Episode, trace_max_chars: int = 8000) -> str:
    """Extract and concatenate CC Tool Activity blocks, truncated."""
    if episode.cc_tool_blocks:
        trace = "\n\n".join(episode.cc_tool_blocks)
    elif episode.mesh_tool_lines:
        trace = "\n".join(episode.mesh_tool_lines)
    else:
        trace = "(no tool activity recorded)"

    if len(trace) > trace_max_chars:
        trace = trace[:trace_max_chars] + "\n... (trace truncated)"
    return trace


def build_outcome(episode: Episode, max_chars: int = 2000) -> str:
    """Extract last agent response, truncated."""
    if not episode.responses:
        return "(no response recorded)"
    last = episode.responses[-1]
    if len(last) > max_chars:
        return last[:max_chars] + "..."
    return last


def get_auto_tags(episode: Episode) -> list[str]:
    """Generate auto-tags based on significance reason."""
    tags = []
    if episode.significance_reason in ("extended-discussion", "brainstorm", "both"):
        if episode.significance_reason == "brainstorm":
            tags.append("brainstorm")
        if episode.significance_reason == "extended-discussion":
            tags.append("extended-discussion")
    return tags


# ── Reflection generation ──────────────────────────────────────────────

async def run_reflection(
    trigger: str,
    trace: str,
    outcome: str,
    llm_client: LLMClient,
) -> ReflectionResult | None:
    """Call LLM with REFLECTION_PROMPT, parse XML response."""
    prompt = REFLECTION_PROMPT.format(
        trigger=trigger,
        trace=trace,
        outcome=outcome[:2000],
    )
    try:
        response = await llm_client.complete(prompt)
    except Exception:
        logger.error("LLM reflection call failed", exc_info=True)
        return None

    reflection = _extract_tag(response, "reflection")
    summary = _extract_tag(response, "summary")
    tags_str = _extract_tag(response, "tags")
    outcome_label = _extract_tag(response, "outcome_label")
    retrieval_key = _extract_tag(response, "retrieval_key")

    if not reflection or not summary:
        logger.warning("Reflection produced empty output")
        return None

    # Fallback: use summary as retrieval key if LLM didn't produce one
    if not retrieval_key:
        retrieval_key = summary

    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
    outcome_label = outcome_label.strip() if outcome_label else "success"
    if outcome_label not in ("success", "partial", "failure"):
        outcome_label = "success"

    return ReflectionResult(
        summary=summary,
        reflection=reflection,
        tags=tags,
        outcome_label=outcome_label,
        retrieval_key=retrieval_key,
    )


# ── Deduplication ──────────────────────────────────────────────────────

def check_dedup(
    candidate_emb: np.ndarray,
    existing_embs: list[np.ndarray],
    threshold: float = 0.9,
) -> tuple[bool, float]:
    """
    Check if a candidate is too similar to existing memories.
    Returns (is_duplicate, max_similarity).
    """
    if not existing_embs:
        return False, 0.0
    max_sim = max(cosine_sim(candidate_emb, emb) for emb in existing_embs)
    return max_sim >= threshold, max_sim


# ── Dry-run reporting ──────────────────────────────────────────────────

def print_dry_run(episodes: list[Episode], total_episodes: int) -> None:
    """Print dry-run report of candidate episodes."""
    tool_heavy = sum(1 for e in episodes if e.significance_reason == "tool-heavy")
    discussion = sum(1 for e in episodes if e.significance_reason == "extended-discussion")
    brainstorm = sum(1 for e in episodes if e.significance_reason == "brainstorm")
    both = sum(1 for e in episodes if e.significance_reason == "both")

    for ep in episodes:
        ts = ep.timestamp[:19] if ep.timestamp else "unknown"
        trigger_preview = ep.trigger[:100].replace("\n", " ")
        if len(ep.trigger) > 100:
            trigger_preview += "..."

        print(f"\nEpisode {ep.index} [{ts}]")
        print(f"  Trigger: \"{trigger_preview}\"")

        if ep.significance_reason == "tool-heavy":
            print(f"  Tool calls: {ep.tool_count} ({ep.tool_summary})")
            resp_chars = sum(len(r) for r in ep.responses)
            print(f"  Response: {resp_chars} chars")
            print(f"  -> CANDIDATE (tool-heavy: {ep.tool_count} tools)")

        elif ep.significance_reason == "extended-discussion":
            print(f"  Tool calls: {ep.tool_count} | Turns: {ep.num_turns} | "
                  f"Chars: {ep.total_chars}")
            print(f"  Agent response: {ep.agent_response_chars} chars")
            print(f"  -> CANDIDATE (extended-discussion: {ep.num_turns} turns, "
                  f"{ep.total_chars} chars)")
            auto_tags = get_auto_tags(ep)
            if auto_tags:
                print(f"  Auto-tags: {', '.join(auto_tags)}")

        elif ep.significance_reason == "brainstorm":
            print(f"  Tool calls: {ep.tool_count} | Turns: {ep.num_turns} | "
                  f"Chars: {ep.total_chars}")
            print(f"  Agent response: {ep.agent_response_chars} chars")
            print(f"  -> CANDIDATE (brainstorm: {ep.num_turns} turns, "
                  f"{ep.agent_response_chars} agent chars, {ep.tool_count} tools)")
            auto_tags = get_auto_tags(ep)
            if auto_tags:
                print(f"  Auto-tags: {', '.join(auto_tags)}")

        elif ep.significance_reason == "both":
            print(f"  Tool calls: {ep.tool_count} ({ep.tool_summary}) | "
                  f"Turns: {ep.num_turns} | Chars: {ep.total_chars}")
            print(f"  Agent response: {ep.agent_response_chars} chars")
            print(f"  -> CANDIDATE (both: {ep.tool_count} tools + "
                  f"{ep.num_turns} turns, {ep.total_chars} chars)")
            auto_tags = get_auto_tags(ep)
            if auto_tags:
                print(f"  Auto-tags: {', '.join(auto_tags)}")

    print(f"\n{'='*60}")
    print(f"Summary: {len(episodes)} candidates out of {total_episodes} episodes.")
    print(f"  Tool-heavy: {tool_heavy}")
    print(f"  Extended discussions: {discussion}")
    print(f"  Brainstorm: {brainstorm}")
    print(f"  Both: {both}")
    print(f"Would create up to {len(episodes)} new memories.")


# ── Live mode ──────────────────────────────────────────────────────────

async def run_live(
    episodes: list[Episode],
    nickname: str,
    llm_client: LLMClient,
    embedder: EmbeddingClient,
    batch_size: int = 10,
    batch_delay: float = 1.0,
    dedup_threshold: float = 0.9,
) -> None:
    """Run live backfill: reflect, embed, deduplicate, store."""
    store = MemoryStore(nickname)
    existing_entries = store.load()
    existing_reflection_embs = [
        e.reflection_embedding for e in existing_entries
        if e.reflection_embedding is not None
    ]

    added = 0
    skipped_dedup = 0
    skipped_error = 0
    total = len(episodes)

    for batch_start in range(0, total, batch_size):
        batch = episodes[batch_start:batch_start + batch_size]

        for i, ep in enumerate(batch):
            ep_num = batch_start + i + 1
            ts = ep.timestamp[:19] if ep.timestamp else "unknown"
            trigger_preview = ep.trigger[:80].replace("\n", " ")
            print(f"\nProcessing episode {ep_num}/{total} [{ts}]...")
            print(f"  Trigger: \"{trigger_preview}\"")

            # Build trace and outcome
            trace = build_trace(ep)
            outcome = build_outcome(ep)

            # Run reflection
            result = await run_reflection(ep.trigger, trace, outcome, llm_client)
            if result is None:
                print(f"  -> SKIPPED (reflection failed)")
                skipped_error += 1
                continue

            # Merge auto-tags
            all_tags = list(set(result.tags + get_auto_tags(ep)))
            print(f"  Reflection: {result.outcome_label} | Tags: {', '.join(all_tags)}")

            # Compute embeddings (reflection + retrieval key)
            try:
                embs = await embedder.embed_batch_to_arrays(
                    [result.reflection, result.retrieval_key]
                )
                reflection_emb = embs[0]
                retrieval_key_emb = embs[1]
            except Exception:
                logger.error("Embedding failed", exc_info=True)
                print(f"  -> SKIPPED (embedding failed)")
                skipped_error += 1
                continue

            print(f"  Embedding computed ({reflection_emb.shape[0]}-dim)")

            # Deduplication check
            is_dup, max_sim = check_dedup(
                reflection_emb, existing_reflection_embs, dedup_threshold
            )
            if is_dup:
                print(f"  Dedup check: similar entry found (sim={max_sim:.3f})")
                print(f"  -> SKIPPED (duplicate)")
                skipped_dedup += 1
                continue

            print(f"  Dedup check: no similar entries (max_sim={max_sim:.3f})")

            # Parse timestamp for created_at
            try:
                created_at = datetime.fromisoformat(
                    ep.timestamp.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                created_at = datetime.now(timezone.utc)

            # Build and store entry
            entry = MemoryEntry(
                id=MemoryEntry.new_id(),
                created_at=created_at,
                summary=result.summary,
                reflection=result.reflection,
                trace=trace,
                trigger=ep.trigger,
                retrieval_key=result.retrieval_key,
                tags=all_tags,
                outcome=result.outcome_label,
                reflection_embedding=reflection_emb,
                retrieval_key_embedding=retrieval_key_emb,
                weight=0.0,
            )
            store.insert(entry)
            existing_reflection_embs.append(reflection_emb)
            added += 1
            print(f"  -> ADDED (memory ID: {entry.id})")

        # Batch delay
        if batch_start + batch_size < total:
            print(f"\n  [batch pause {batch_delay}s...]")
            await asyncio.sleep(batch_delay)

    # Recompute active set at the end
    print(f"\nRecomputing active set...")
    all_entries = store.load()
    embs_with_idx = [
        (i, e.reflection_embedding)
        for i, e in enumerate(all_entries)
        if e.reflection_embedding is not None
    ]
    if embs_with_idx:
        emb_list = [emb for _, emb in embs_with_idx]
        selected_indices, weights, f_s = select_active_set(emb_list, 30)

        # Update weights in store
        weight_updates = {}
        for i, e in enumerate(all_entries):
            weight_updates[e.id] = 0.0
        for sel_pos, emb_idx in enumerate(selected_indices):
            pool_idx = embs_with_idx[emb_idx][0]
            entry = all_entries[pool_idx]
            weight_updates[entry.id] = weights[sel_pos]
        store.update_weights_batch(weight_updates)
        print(f"Active set: {len(selected_indices)} entries selected from pool of {len(all_entries)}")

    store.close()

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Episodes processed: {total}")
    print(f"  New memories added: {added}")
    print(f"  Skipped (dedup): {skipped_dedup}")
    print(f"  Skipped (error): {skipped_error}")


# ── Main ───────────────────────────────────────────────────────────────

async def backfill(args: argparse.Namespace) -> None:
    """Main entry point: parse, filter, reflect, deduplicate, store."""
    print(f"Loading history for agent: {args.nickname}")
    history = load_history(args.nickname)
    print(f"Loaded {len(history)} history entries")

    print(f"Parsing episodes (session gap: {args.session_gap}s)...")
    episodes = parse_episodes(history, session_gap_secs=args.session_gap)
    total_episodes = len(episodes)
    print(f"Found {total_episodes} episodes")

    print(f"Filtering (min-tools={args.min_tools}, "
          f"min-discussion-turns={args.min_discussion_turns}, "
          f"min-discussion-chars={args.min_discussion_chars}, "
          f"max-brainstorm-tools={args.max_brainstorm_tools}, "
          f"min-brainstorm-response-chars={args.min_brainstorm_response_chars}"
          f"{f', since={args.since}' if args.since else ''})...")
    candidates = filter_episodes(
        episodes,
        min_tools=args.min_tools,
        min_discussion_turns=args.min_discussion_turns,
        min_discussion_chars=args.min_discussion_chars,
        max_brainstorm_tools=args.max_brainstorm_tools,
        min_brainstorm_response_chars=args.min_brainstorm_response_chars,
        since=args.since,
    )
    print(f"Found {len(candidates)} significant episodes")

    if args.max_episodes and len(candidates) > args.max_episodes:
        candidates = candidates[:args.max_episodes]
        print(f"Capped at {args.max_episodes} episodes")

    if not candidates:
        print("No significant episodes found. Nothing to do.")
        return

    if args.dry_run:
        print_dry_run(candidates, total_episodes)
        return

    # Live mode: set up LLM client and embedder
    backend = args.backend or "openai"
    model = args.model or "gpt-5.1"
    llm_config = LLMConfig.from_env(backend=backend)
    llm_config.model = model
    llm_config.max_tokens = 4096
    llm_config.temperature = 0.7

    embedder = EmbeddingClient(
        backend="openai",
        model="text-embedding-3-small",
    )

    async with LLMClient(llm_config) as llm_client:
        await run_live(
            candidates,
            nickname=args.nickname,
            llm_client=llm_client,
            embedder=embedder,
            batch_size=args.batch_size,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Backfill memory pool from agent conversation history"
    )
    parser.add_argument("--agent", required=True, help="Agent type (coder, sysadmin)")
    parser.add_argument("--nickname", required=True, help="Agent nickname")
    parser.add_argument("--min-tools", type=int, default=3,
                        help="Min tool calls for tool-heavy criterion (default: 3)")
    parser.add_argument("--min-discussion-turns", type=int, default=4,
                        help="Min turns for extended-discussion criterion (default: 4)")
    parser.add_argument("--min-discussion-chars", type=int, default=1500,
                        help="Min total chars for extended-discussion criterion (default: 1500)")
    parser.add_argument("--max-brainstorm-tools", type=int, default=2,
                        help="Max tool calls for brainstorm criterion (default: 2)")
    parser.add_argument("--min-brainstorm-response-chars", type=int, default=1500,
                        help="Min agent response chars for brainstorm criterion (default: 1500)")
    parser.add_argument("--session-gap", type=int, default=900,
                        help="Min seconds between user messages to start a new episode (default: 900 = 15min)")
    parser.add_argument("--since", type=str, default=None,
                        help="Only process episodes after this date (YYYY-MM-DD)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Cap on episodes to process")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Episodes per batch (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show candidates without LLM calls or DB writes")
    parser.add_argument("--backend", type=str, default=None,
                        help="LLM backend for reflections (default: openai)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model for reflections (default: gpt-5.1)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    asyncio.run(backfill(args))


if __name__ == "__main__":
    main()
