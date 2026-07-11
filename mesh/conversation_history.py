"""
Shared conversation history component.

Provides summary + recent-window context management for both
worker (AgentNode) and router (RouterV2).

Design:
- Absorbs the role of SummaryState for summary management
- Token estimation delegates to existing mesh.llm functions
- Monotonic seq_id per Turn for stable references (replaces positional offsets)
- Persistence: v2 format with auto-detection of v1 flat lists
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from .llm import estimate_tokens, estimate_history_tokens, HistoryMessage, SUMMARIZATION_PROMPT
from .node import SummaryState

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)


# Router-specific summarization prompt (incremental, topic-aware)
ROUTER_SUMMARY_PROMPT = """You are extending a message router's conversation summary with a new batch of turns.

Below you will see an existing summary (if any) and new turns to incorporate.

Focus on information relevant to ROUTING future messages:
- Which agents handled which tasks and the outcomes
- Ongoing multi-step tasks and their current state
- User preferences for agent assignment
- Recent routing decisions and their rationale

Organize by topic with timestamps and status ([COMPLETED] / [ACTIVE] / [PENDING]).
Compress old completed topics to single lines. Keep active topics detailed.

CONVERSATION TO SUMMARIZE:

{span_text}

Provide your updated routing summary."""


@dataclass
class Turn:
    """A single conversational turn in the history window.

    Convention for trace-as-history (Turn.meta keys; see
    docs/plans/trace-as-history-2026-04-27.md):

      trace_block:    "tool_call" | "tool_result"  (presence = trace Turn)
      tool_name:      str  (e.g. "memory_get")
      tool_call_id:   str  (links a tool_call/tool_result pair)
      tool_args:      dict  (parsed args from a tool_call; optional)
      tool_success:   bool  (tool_result only)
      truncated:      bool  (tool_result only; True if line/char cap fired)
      original_lines: int   (tool_result only; pre-truncation line count)
      original_chars: int   (tool_result only; pre-truncation char count)
      worker_id:      str   (the dispatch that generated this Turn)
    """
    role: str                    # "user", "assistant", "system", "router", "tool"
    content: str
    timestamp: datetime | str
    from_node: str = ""          # source node ID (for XML rendering)
    to_node: str | None = None   # destination node ID
    meta: dict[str, Any] = field(default_factory=dict)
    seq_id: int = 0              # monotonic sequence ID, assigned by ConversationHistory
    _token_estimate: int | None = field(default=None, repr=False)

    @property
    def token_estimate(self) -> int:
        if self._token_estimate is None:
            self._token_estimate = estimate_tokens(self.content)
        return self._token_estimate

    def to_history_message(self, source: str = "persisted") -> HistoryMessage:
        """Convert to HistoryMessage for compatibility with existing LLM pipeline."""
        ts = self.timestamp
        if isinstance(ts, datetime):
            ts = ts.isoformat()
        return HistoryMessage(
            from_node=self.from_node or self.role,
            content=self.content,
            timestamp=ts,
            to_node=self.to_node,
            source=source,
        )

    def to_dict(self) -> dict:
        ts = self.timestamp
        if isinstance(ts, datetime):
            ts = ts.isoformat()
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": ts,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "meta": self.meta,
            "seq_id": self.seq_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Turn:
        ts = data.get("timestamp", "")
        if isinstance(ts, str) and ts:
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                pass  # keep as string
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=ts,
            from_node=data.get("from_node", ""),
            to_node=data.get("to_node"),
            meta=data.get("meta", {}),
            seq_id=data.get("seq_id", 0),
        )

    def is_trace_block(self) -> bool:
        """True if this Turn is a tool-call or tool-result entry."""
        return self.meta.get("trace_block") in ("tool_call", "tool_result")


class ConversationHistory:
    """Shared summary + recent-window conversation history.

    Used by both worker (AgentNode) and router (RouterV2).
    Absorbs the role of SummaryState for summary management.
    """

    def __init__(
        self,
        *,
        soft_token_limit: int = 50_000,
        hard_token_limit: int = 150_000,
        target_ratio: float = 0.25,
        base_overhead: int = 8_500,
        per_message_overhead: int = 35,
        max_summarization_tokens: int = 80_000,
        summarization_prompt: str | None = None,
        persist_path: Path | None = None,
        summary_persist_path: Path | None = None,
        summarization_enabled: bool = True,
        window_budget: int | None = None,
        max_summary_tokens: int | None = None,
    ):
        # --- Summary state (replaces SummaryState) ---
        self._summary: SummaryState | None = None

        # --- Recent window ---
        self._window: list[Turn] = []
        self._next_seq_id: int = 1  # monotonic counter

        # --- Rolling window budget (W) ---
        # If window_budget is explicitly set, use it.
        # Otherwise derive from soft_token_limit for backward compatibility:
        #   W = soft_limit // 2  →  trigger at 2W = soft_limit
        self._window_budget = window_budget if window_budget is not None else (soft_token_limit // 2)

        # --- Summary token cap ---
        # Prevents unbounded summary growth over many summarization cycles.
        # Default: W // 4 (~8.75K for W=35K). If the summary exceeds this,
        # the oldest content is truncated.
        self._max_summary_tokens = max_summary_tokens if max_summary_tokens is not None else (self._window_budget // 4)

        # --- Limits (kept for backward compat / hard limit safety net) ---
        self._soft_limit = soft_token_limit
        self._hard_limit = hard_token_limit
        self._target = int(soft_token_limit * target_ratio)
        self._target_ratio = target_ratio
        self._base_overhead = base_overhead
        self._per_msg_overhead = per_message_overhead
        self._max_summarization_tokens = max_summarization_tokens

        # --- Summarization ---
        self._summarization_prompt = summarization_prompt or SUMMARIZATION_PROMPT
        self._summarizing = False
        self._summarization_task: asyncio.Task | None = None

        # --- Summarization enabled flag ---
        self._summarization_enabled = summarization_enabled

        # --- Persistence ---
        self._persist_path = persist_path
        self._summary_persist_path = summary_persist_path or (
            Path(str(persist_path).replace(".json", ".summary.json"))
            if persist_path else None
        )

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def summary(self) -> SummaryState | None:
        """The current summary state, if any."""
        return self._summary

    @summary.setter
    def summary(self, value: SummaryState | None) -> None:
        self._summary = value

    @property
    def window(self) -> list[Turn]:
        """The recent window of turns."""
        return self._window

    @property
    def is_summarizing(self) -> bool:
        """Whether summarization is currently in progress."""
        return self._summarizing

    @property
    def window_budget(self) -> int:
        """The token budget for the rolling window."""
        return self._window_budget

    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------

    def append(self, turn: Turn) -> None:
        """Add a turn to the recent window. Assigns a monotonic seq_id."""
        turn.seq_id = self._next_seq_id
        self._next_seq_id += 1
        self._window.append(turn)

    @property
    def last_seq_id(self) -> int:
        """The seq_id of the most recent turn, or 0 if empty."""
        return self._window[-1].seq_id if self._window else 0

    def turns_since(self, seq_id: int) -> list[Turn]:
        """Return all turns with seq_id > the given value.

        Used by router to track "what's new" from a worker's history
        without fragile positional offsets.
        """
        return [t for t in self._window if t.seq_id > seq_id]

    def iter_tool_results(self, tool_name: str | None = None):
        """Iterate window Turns that are tool-result entries.

        Used by the retrieval-plan TOC dedup. Filters by tool_name if given.
        Turns missing meta or the trace_block key are skipped silently.
        """
        for t in self._window:
            if t.meta.get("trace_block") != "tool_result":
                continue
            if tool_name is not None and t.meta.get("tool_name") != tool_name:
                continue
            yield t

    def get_recent_for_peek(self, n: int = 20, since_seq: int = 0) -> list[Turn]:
        """Return recent turns for external peeking (e.g., router peeking at worker).

        If since_seq > 0, return only turns with seq_id > since_seq.
        Otherwise return the last N turns.
        """
        if since_seq > 0:
            return self.turns_since(since_seq)
        return self._window[-n:]

    def __len__(self) -> int:
        return len(self._window)

    # -------------------------------------------------------------------------
    # Token estimation — delegates to existing mesh.llm functions
    # -------------------------------------------------------------------------

    def estimate_tokens(self) -> int:
        """Estimate total tokens: base_overhead + summary + window.

        Uses the same formula as estimate_history_tokens() in mesh/llm.py:
            base_overhead + Σ(estimate_tokens(msg.content) + per_msg_overhead)
        """
        total = self._base_overhead

        if self._summary:
            total += self._summary.token_estimate

        for turn in self._window:
            total += turn.token_estimate + self._per_msg_overhead

        return total

    def estimate_window_tokens(self) -> int:
        """Estimate tokens in the rolling window only (excludes summary and overhead)."""
        total = 0
        for turn in self._window:
            total += turn.token_estimate + self._per_msg_overhead
        return total

    # -------------------------------------------------------------------------
    # Summarization
    # -------------------------------------------------------------------------

    def partition_and_drop_old(self) -> list[Turn]:
        """Partition window at the W-token boundary and drop old half.

        Used by memory v2: no summary generated, just returns the old turns
        and trims the window to the recent half.

        Returns the old_half turns (empty list if nothing to split).
        """
        if self._summarizing:
            return []

        self._summarizing = True
        try:
            W = self._window_budget
            accumulated = 0
            partition_idx = 0
            for i, turn in enumerate(self._window):
                cost = turn.token_estimate + self._per_msg_overhead
                if accumulated + cost > W:
                    partition_idx = i
                    break
                accumulated += cost
                partition_idx = i + 1

            if partition_idx <= 0:
                if len(self._window) > 1:
                    partition_idx = 1
                else:
                    return []

            old_half = self._window[:partition_idx]
            recent_half = self._window[partition_idx:]

            self._window = recent_half
            logger.info(
                "Window partition (v2): dropped %d turns (~%d tokens), "
                "kept %d turns (~%d tokens)",
                len(old_half), accumulated,
                len(recent_half), self.estimate_window_tokens(),
            )
            self._save_if_configured()
            return old_half
        finally:
            self._summarizing = False

    def needs_summarization(self) -> bool:
        """True if the rolling window has grown to 2× the window budget (W).

        Rolling window model:
          - State at rest: [summary] + [~W tokens of recent turns]
          - Trigger: window grows to 2W
          - Action: compress oldest W tokens into summary, keep recent W
        """
        if self._summarizing:
            window_tokens = self.estimate_window_tokens()
            if window_tokens >= 2 * self._window_budget:
                logger.warning(
                    f"Window at {window_tokens} tokens (>= 2×W={2 * self._window_budget}) "
                    f"but summarization already in progress — window growing unbounded"
                )
            return False
        return self.estimate_window_tokens() >= 2 * self._window_budget

    async def summarize(
        self,
        llm_client: "LLMClient",
        model: str | None = None,
        format_fn: Callable[[list[Turn]], str] | None = None,
    ) -> None:
        """Rolling window summarization: compress the oldest W tokens of the window.

        Algorithm:
        1. Walk window front-to-back, accumulating tokens until reaching W → old_half.
        2. Remainder → recent_half.
        3. LLM(existing_summary + old_half) → new_summary.
        4. Discard old_half. Window = recent_half.
        5. If summary exceeds max_summary_tokens, truncate oldest content.

        This produces small, regular compressions instead of the cliff-edge
        behavior of the old algorithm (which compressed 75% of the window at once).

        In the AgentNode path, _run_summarization() overrides messages_summarized
        with the _history-indexed value. ConversationHistory.summarize() should not
        rely on its internal counter for correctness when used via AgentNode.

        Args:
            llm_client: LLMClient instance (can be a dedicated summarization client)
            model: Optional model override for the summarization call
            format_fn: Optional custom history formatter (default: _default_format)
        """
        if self._summarizing:
            return
        self._summarizing = True

        try:
            W = self._window_budget

            # Partition window at the W-token boundary
            accumulated = 0
            partition_idx = 0
            for i, turn in enumerate(self._window):
                cost = turn.token_estimate + self._per_msg_overhead
                if accumulated + cost > W:
                    partition_idx = i
                    break
                accumulated += cost
                partition_idx = i + 1

            if partition_idx <= 0:
                if len(self._window) > 1:
                    # Edge case: first turn alone exceeds W.
                    # Fold it into the summary anyway to make progress.
                    partition_idx = 1
                    accumulated = self._window[0].token_estimate + self._per_msg_overhead
                else:
                    return  # single turn, nothing to split

            # Snapshot the window length before the (slow) LLM call.
            # Entries appended to self._window while we await must be
            # preserved — see "TOCTOU race" below.
            pre_summarize_len = len(self._window)

            old_half = self._window[:partition_idx]
            recent_half = self._window[partition_idx:pre_summarize_len]

            # Build input text for summarization
            formatter = format_fn or self._default_format
            parts = []
            if self._summary:
                parts.append(f"[Existing summary]\n{self._summary.summary_text}\n")
            parts.append("[New turns to incorporate]\n" + formatter(old_half))
            text = "\n".join(parts)

            # Bootstrap: truncate if text is too large for the summarization model
            text_tokens = estimate_tokens(text)
            if text_tokens > self._max_summarization_tokens:
                char_budget = self._max_summarization_tokens * 4
                text = text[-char_budget:]
                logger.info(
                    f"Bootstrap truncation: {text_tokens} tokens -> "
                    f"~{self._max_summarization_tokens} tokens (char budget {char_budget})"
                )

            # Call LLM using the summarization prompt
            prompt = self._summarization_prompt.format(span_text=text)
            summary_text = await llm_client.complete(prompt, model=model)

            # Enforce summary token cap to prevent unbounded growth
            summary_token_count = estimate_tokens(summary_text)
            if self._max_summary_tokens > 0 and summary_token_count > self._max_summary_tokens:
                logger.warning(
                    f"Summary exceeds cap ({summary_token_count} > {self._max_summary_tokens} tokens), "
                    f"truncating oldest content"
                )
                # Truncate from the beginning (oldest content) to fit the cap
                char_budget = self._max_summary_tokens * 4
                if len(summary_text) > char_budget:
                    summary_text = "...\n" + summary_text[-char_budget:]
                    summary_token_count = estimate_tokens(summary_text)

            # Build new SummaryState
            summarize_count = len(old_half)
            messages_summarized = (
                (self._summary.messages_summarized if self._summary else 0) + summarize_count
            )
            self._summary = SummaryState(
                summary_text=summary_text,
                messages_summarized=messages_summarized,
                created_at=datetime.now(timezone.utc).isoformat(),
                token_estimate=summary_token_count,
            )

            # TOCTOU race fix: entries may have been appended to self._window
            # while we awaited the LLM call.  Capture and re-attach them so
            # they aren't silently dropped when we replace the window.
            new_during_summarize = self._window[pre_summarize_len:]
            self._window = recent_half + new_during_summarize

            if new_during_summarize:
                logger.info(
                    f"Rolling window summarization: rescued {len(new_during_summarize)} "
                    f"entries appended during LLM call"
                )

            logger.info(
                f"Rolling window summarization: folded {summarize_count} turns "
                f"(~{accumulated} tokens) into summary "
                f"(~{summary_token_count} tokens), "
                f"kept {len(self._window)} recent turns "
                f"(~{self.estimate_window_tokens()} tokens)"
            )

            # Persist
            self._save_if_configured()

        except Exception as e:
            logger.exception(f"Summarization failed: {e}")
        finally:
            self._summarizing = False

    @staticmethod
    def _default_format(turns: list[Turn]) -> str:
        """Format turns for summarization, grouped by topic label when available.

        Turns with topic labels are grouped under section headers:
          --- Topic: document editing (2026-02-26 22:26) ---
          [user:yourname at ...] ...

        Turns without topic labels are rendered inline without headers.
        Consecutive turns with the same topic share a single header.
        """
        lines = []
        current_topic = None
        for t in turns:
            topic = t.meta.get("topic_label", "")
            if topic and topic != current_topic:
                # Emit topic section header
                ts = t.timestamp
                if isinstance(ts, datetime):
                    ts_str = ts.strftime("%Y-%m-%d %H:%M")
                else:
                    ts_str = str(ts)[:16]
                lines.append(f"\n--- Topic: {topic} ({ts_str}) ---")
                current_topic = topic
            elif not topic and current_topic:
                current_topic = None

            label = t.from_node or t.role
            ts = t.timestamp
            if isinstance(ts, datetime):
                ts = ts.isoformat()
            lines.append(f"[{label} at {ts}]")
            lines.append(t.content)
            lines.append("")  # Blank line between messages
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Build context for LLM
    # -------------------------------------------------------------------------

    def build_context_for_llm(self) -> list[HistoryMessage]:
        """Return summary (as system message) + recent window as HistoryMessage list.

        Behavior-equivalent to _build_history_for_llm() in agent_node.py:
        - Summary injected as a system message with "[Earlier summary]" prefix.
        - Window turns converted to HistoryMessage.
        - Hard cap applied: if total exceeds hard_limit, drop oldest window turns.
        """
        result: list[HistoryMessage] = []

        # Inject summary as system message
        if self._summary and self._summary.summary_text:
            result.append(HistoryMessage(
                from_node="system",
                content=f"[Earlier summary]\n{self._summary.summary_text}",
                timestamp=self._summary.created_at,
                source="persisted",
            ))

        # Add window turns
        for turn in self._window:
            result.append(turn.to_history_message())

        # Hard cap: drop oldest window entries if over limit
        # Pre-compute per-entry token costs to avoid O(n²) re-estimation
        entry_tokens = [estimate_tokens(msg.content) + 35 for msg in result]
        total = self._base_overhead + sum(entry_tokens)
        has_summary = bool(self._summary and self._summary.summary_text)
        drop_start = 1 if has_summary else 0  # don't drop the summary message
        dropped = 0
        while total > self._hard_limit and len(entry_tokens) > drop_start + 1:
            total -= entry_tokens.pop(drop_start)
            result.pop(drop_start)
            dropped += 1

        # Prune self._window so dropped entries don't return next call
        if dropped > 0:
            self._window = self._window[dropped:]
            logger.info(
                f"Hard limit prune: dropped {dropped} oldest window entries "
                f"({len(self._window)} remaining, ~{total} tokens)"
            )
            self.save()

        return result

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """Persist window + summary to disk."""
        if self._persist_path:
            data = {
                "version": 2,
                "next_seq_id": self._next_seq_id,
                "window": [t.to_dict() for t in self._window],
            }
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps(data, indent=2))

        if self._summary_persist_path and self._summary:
            self._summary_persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._summary_persist_path.write_text(
                json.dumps(self._summary.to_dict(), indent=2)
            )

    def load(self) -> int:
        """Load window + summary from disk. Returns count of entries loaded.

        Handles both formats:
        - v2 (new): {"version": 2, "next_seq_id": N, "window": [...]}
        - v1 (legacy flat): [{"role": ..., "content": ...}, ...]
          -> Treated as: summary=None, entire list becomes window.
        """
        loaded = 0

        if self._persist_path and self._persist_path.exists():
            try:
                raw = json.loads(self._persist_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Failed to load history from {self._persist_path}: {e}")
                return 0

            if isinstance(raw, dict) and raw.get("version") == 2:
                # v2 format
                self._window = [Turn.from_dict(d) for d in raw.get("window", [])]
                self._next_seq_id = raw.get("next_seq_id", len(self._window) + 1)
            elif isinstance(raw, list):
                # v1 legacy format: flat list of dicts (HistoryEntry format)
                self._window = []
                for i, entry in enumerate(raw):
                    # v1 entries are HistoryEntry dicts with message/direction
                    if "message" in entry:
                        msg_data = entry["message"]
                        content = msg_data.get("content", "")
                        if isinstance(content, dict):
                            content = str(content)
                        ts = msg_data.get("timestamp", "")
                        from_node = msg_data.get("from_node", "")
                        to_node = msg_data.get("to_node")
                        role = entry.get("direction", "user")
                    else:
                        # Simple dict format
                        content = entry.get("content", "")
                        ts = entry.get("timestamp", "")
                        from_node = entry.get("from_node", entry.get("role", ""))
                        to_node = entry.get("to_node")
                        role = entry.get("role", "user")

                    if isinstance(ts, str) and ts:
                        try:
                            ts = datetime.fromisoformat(ts)
                        except ValueError:
                            ts = datetime.now(timezone.utc)
                    elif not ts:
                        ts = datetime.now(timezone.utc)

                    self._window.append(Turn(
                        role=role,
                        content=content,
                        timestamp=ts,
                        from_node=from_node,
                        to_node=to_node,
                        seq_id=i + 1,
                    ))
                self._next_seq_id = len(self._window) + 1
            else:
                logger.warning(f"Unknown history format in {self._persist_path}")
                return 0

            loaded = len(self._window)

        # Load summary (same format as existing SummaryState)
        # Skip if summarization is disabled — old summaries would consume tokens indefinitely
        if self._summarization_enabled and self._summary_persist_path and self._summary_persist_path.exists():
            try:
                summary_data = json.loads(self._summary_persist_path.read_text())
                self._summary = SummaryState.from_dict(summary_data)
                logger.info(
                    f"Loaded summary: {self._summary.messages_summarized} messages, "
                    f"~{self._summary.token_estimate} tokens"
                )
            except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
                logger.error(f"Failed to load summary: {e}")
        elif not self._summarization_enabled:
            self._summary = None

        return loaded

    def _save_if_configured(self) -> None:
        """Auto-save after summarization if persistence is configured."""
        if self._persist_path or self._summary_persist_path:
            self.save()

    # -------------------------------------------------------------------------
    # Serialization (for programmatic use / tests)
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "next_seq_id": self._next_seq_id,
            "summary": self._summary.to_dict() if self._summary else None,
            "window": [t.to_dict() for t in self._window],
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> ConversationHistory:
        hist = cls(**kwargs)
        if data.get("summary"):
            hist._summary = SummaryState.from_dict(data["summary"])
        hist._window = [Turn.from_dict(d) for d in data.get("window", [])]
        hist._next_seq_id = data.get("next_seq_id", len(hist._window) + 1)
        return hist
