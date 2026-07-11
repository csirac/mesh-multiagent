"""Memory Formation v3 — structured-JSON LLM segmenter.

Port of the LLMSegmenter prototype from benchmark/memory_dryrun/strategies.py
adapted for the live LLM client (`self._llm_client.complete` async) and the
mesh ConversationHistory `Turn` shape.

See docs/plans/memory-formation-v3-2026-04-27.md (rev 6) §2.2-§2.4.

One LLM call per window produces structured per-segment metadata:
    {worthwhile, score, retrieval_key, summary, tags, outcome, topic_label, project}

Window stride is `window_size - overlap`. Trailing `defer_tail` turns within
a non-final window are deferred — the next window re-classifies with full
forward context. The final window emits everything (no future context to
wait for). Already-emitted ranges are tracked so overlap windows never
double-emit.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A contiguous run of turns sharing a topic/intent.

    `metadata` carries optional structured per-segment data:
    `worthwhile` (bool), `score` (int 0-10), `retrieval_key` (str),
    `summary` (str), `tags` (list[str]), `outcome` (success|partial|failure|None),
    `project` (str|None — segmenter-suggested project name for new-project bootstrap).
    """

    turns: list[Any]
    topic_label: str = ""
    start_idx: int = 0  # global index in the input turns list
    end_idx: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _turn_summary(turn: Any, max_chars: int = 160) -> str:
    """Render a Turn as a short single-line summary for the segmenter prompt."""
    role = getattr(turn, "role", "") or ""
    from_node = getattr(turn, "from_node", "") or ""
    to_node = getattr(turn, "to_node", "") or ""
    content = getattr(turn, "content", "") or ""
    content_str = content if isinstance(content, str) else str(content)
    # Strip newlines for a single-line summary
    content_oneline = " ".join(content_str.split())
    if len(content_oneline) > max_chars:
        content_oneline = content_oneline[:max_chars] + "..."
    if from_node.startswith("user:"):
        prefix = f"USER {from_node.split(':', 1)[-1]}"
    elif from_node.startswith("agent:"):
        prefix = f"AGENT {from_node.split(':')[-1]}"
    elif from_node.startswith("channel:"):
        prefix = f"CHANNEL {from_node.split(':', 1)[-1][:24]}"
    elif role == "tool":
        prefix = "TOOL"
    elif role:
        prefix = role.upper()
    else:
        prefix = "MSG"
    if to_node and to_node != "internal":
        prefix = f"{prefix}->{to_node.split(':')[-1]}"
    return f"[{prefix}] {content_oneline}"


class LLMSegmenterV3:
    """Structured-output LLM segmenter with overlap + defer.

    Each window of `window_size` turns is sent to the LLM in a single
    call returning per-segment metadata. Stride is `window_size - overlap`.
    Trailing `defer_tail` turns are deferred (re-classified by the next
    window) except in the final window, which emits unconditionally.

    Telemetry attributes (read by callers for logging):
        windows_called, json_parse_failures, malformed_segments_dropped,
        deferred_segments
    """

    PROMPT = (
        'You will receive a numbered list of turns from a chat log between a user, agents, and channels.\n'
        'Segment them into discrete topics/episodes and produce structured metadata for each segment.\n'
        '\n'
        '## Turns (0-indexed within this window):\n'
        '{turn_summaries}\n'
        '\n'
        '{project_section}\n'
        '\n'
        '## Task\n'
        'For each contiguous run of turns belonging to one topic/task/episode, emit one segment object.\n'
        'Segments must be contiguous, ordered, and cover indices 0..{last_idx} without gaps.\n'
        'Aim for 1-12 segments depending on actual topic shifts. Single-turn segments are fine\n'
        'when a turn truly stands alone (e.g. an interruption between threads).\n'
        '\n'
        '## Output schema — return ONLY a single JSON object, nothing else:\n'
        '{{"segments": [\n'
        '  {{\n'
        '    "start_turn": <int, 0-based first turn index in this window>,\n'
        '    "end_turn":   <int, 0-based last turn index in this window, inclusive>,\n'
        '    "topic_label": "short topic phrase, max 8 words",\n'
        '    "worthwhile": <bool — true if this segment is worth remembering>,\n'
        '    "score": <int 0-10 — how memorable/important is this episode>,\n'
        '    "retrieval_key": "1-2 sentences describing what to search for to find this. Be specific — name technologies, files, people.",\n'
        '    "summary": "1-3 sentences distilling what happened and the key takeaway.",\n'
        '    "tags": ["domain", "tag", "list"],\n'
        '    "outcome": "success" | "partial" | "failure" | null,\n'
        '    "project": "<exact name from the Known project names list above, or null>"\n'
        '  }},\n'
        '  ...\n'
        ']}}\n'
        '\n'
        '## Rules for the worthwhile flag\n'
        '- worthwhile=false for: trivial pings, routing acks, quick yes/no clarifications,\n'
        '  greetings, single-turn stand-alone messages with no follow-up substance.\n'
        '- worthwhile=true for: substantive task work, debugging, decisions, plans, design,\n'
        '  reflections, anything you would want to recall in a future session.\n'
        '- score reflects intensity within "worthwhile": 0-3 = trivial, 4-6 = routine task,\n'
        '  7-8 = substantial work, 9-10 = key decision or hard-won lesson.\n'
        '\n'
        'Output only the JSON object, no prose, no code fences.\n'
    )

    def __init__(
        self,
        llm_client,
        *,
        window_size: int = 60,
        overlap: int = 20,
        defer_tail_turns: int = 10,
        max_tokens: int = 8000,
        temperature: float = 0.1,
        model: str | None = None,
        request_timeout: float = 240.0,
    ) -> None:
        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must be in [0, window_size)")
        if defer_tail_turns < 0:
            raise ValueError("defer_tail_turns must be >= 0")
        if defer_tail_turns >= window_size:
            raise ValueError("defer_tail_turns must be < window_size")
        stride = window_size - overlap
        if stride <= 0:
            raise ValueError("stride (window_size - overlap) must be > 0")
        self._llm_client = llm_client
        self.window_size = window_size
        self.overlap = overlap
        self.defer_tail_turns = defer_tail_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = model
        self.request_timeout = request_timeout
        # Telemetry
        self.windows_called = 0
        self.json_parse_failures = 0
        self.malformed_segments_dropped = 0
        self.deferred_segments = 0

    @property
    def stride(self) -> int:
        return self.window_size - self.overlap

    # ── LLM call + parse ────────────────────────────────────────────

    @staticmethod
    def _format_known_projects(known_projects: list[str] | str | None) -> str:
        """Render the full project section for the prompt.

        Accepts a list[str], a comma-separated str (matches what
        `MemorySystemV2._known_project_names()` returns), or None/empty.
        Returns the complete '## Known project names' block with
        context-appropriate rules.
        """
        names: list[str] = []
        if isinstance(known_projects, str):
            names = [n.strip() for n in known_projects.split(",") if n.strip()]
        elif known_projects is not None:
            names = [str(n).strip() for n in known_projects if str(n).strip()]

        if not names:
            return (
                '## Project labels\n'
                'No known projects yet. For the "project" field in each segment:\n'
                '- Suggest a concise project label (lowercase, hyphenated, e.g. "memory-system")\n'
                '  that describes the work area.\n'
                '- Use null only if the segment is trivial or a one-off exchange.\n'
            )
        project_list = "\n".join(f"- {n}" for n in names)
        return (
            '## Known project names\n'
            f'{project_list}\n'
            '\n'
            'Rules for the project field:\n'
            '- Pick the matching name from the list above when the segment clearly belongs to a known project.\n'
            '- If no listed project applies, suggest a concise NEW project label\n'
            '  (lowercase, hyphenated, e.g. "recipe-lookup", "travel-planning", "email-automation").\n'
            '  New labels are encouraged — they help organize diverse work.\n'
            '- Use null ONLY for trivial single-turn exchanges (greetings, acknowledgments,\n'
            '  small talk with no substantive content).\n'
        )

    async def _call_llm(
        self,
        window_turns: list[Any],
        known_projects: list[str] | str | None = None,
    ) -> str:
        summaries = "\n".join(
            f"{i}. {_turn_summary(t, max_chars=160)}"
            for i, t in enumerate(window_turns)
        )
        prompt = self.PROMPT.format(
            turn_summaries=summaries,
            last_idx=len(window_turns) - 1,
            project_section=self._format_known_projects(known_projects),
        )
        return await asyncio.wait_for(
            self._llm_client.complete(
                prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
            timeout=self.request_timeout,
        )

    @staticmethod
    def _strip_fences(raw: str) -> str:
        return re.sub(
            r"^```(?:json)?|```$",
            "",
            raw.strip(),
            flags=re.MULTILINE,
        ).strip()

    def _parse_window(self, raw: str) -> list[dict[str, Any]] | None:
        """Parse the LLM's JSON output into a list of segment dicts.

        Returns None on shape failure. Returns a (possibly empty) list of
        dicts otherwise; individual malformed segments are dropped silently
        downstream and counted in `malformed_segments_dropped`.
        """
        try:
            data = json.loads(self._strip_fences(raw))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        segs = data.get("segments")
        if not isinstance(segs, list):
            return None
        return segs

    def _validate_segment_dict(
        self, d: Any, window_len: int
    ) -> dict[str, Any] | None:
        """Validate one raw segment dict. Returns a normalised dict or None."""
        if not isinstance(d, dict):
            return None
        try:
            rs = int(d["start_turn"])
            re_ = int(d["end_turn"])
        except (KeyError, TypeError, ValueError):
            return None
        if rs < 0 or re_ < rs or re_ >= window_len:
            return None
        topic = str(d.get("topic_label", "") or "").strip()[:120] or "untitled"
        worthwhile = bool(d.get("worthwhile", False))
        try:
            score = int(d.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(10, score))
        retrieval_key = str(d.get("retrieval_key", "") or "").strip()
        summary = str(d.get("summary", "") or "").strip()
        tags_raw = d.get("tags") or []
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        elif isinstance(tags_raw, list):
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        else:
            tags = []
        outcome = d.get("outcome")
        if isinstance(outcome, str):
            ol = outcome.strip().lower()
            outcome = ol if ol in ("success", "partial", "failure") else None
        else:
            outcome = None
        project = d.get("project")
        if isinstance(project, str):
            project = project.strip()
            if not project or project.lower() in ("null", "none"):
                project = None
        else:
            project = None
        return {
            "rel_start": rs,
            "rel_end": re_,
            "topic_label": topic,
            "worthwhile": worthwhile,
            "score": score,
            "retrieval_key": retrieval_key,
            "summary": summary,
            "tags": tags,
            "outcome": outcome,
            "project": project,
        }

    # ── Window-stepping with overlap + defer ────────────────────────

    async def segment(
        self,
        turns: list[Any],
        *,
        known_projects: list[str] | str | None = None,
    ) -> list[Segment]:
        """Run the window-stepping segmentation loop.

        Returns a list of Segment objects. Raises on persistent LLM failure
        (caller treats as parse failure for fallback bookkeeping).

        `known_projects` is the canonical list of project names the agent
        already knows about (typically from the project_maps table). The
        segmenter is instructed to pick from this list or use null —
        avoiding hallucinated project names that fragment retrieval.
        Accepts a list[str], a comma-separated str (matching
        `MemorySystemV2._known_project_names()`), or None/empty.
        """
        if not turns:
            return []

        emitted: list[Segment] = []
        N = len(turns)
        frontier = 0
        window_start = 0
        any_window_succeeded = False
        all_windows_failed = True

        while window_start < N:
            window_end = min(window_start + self.window_size, N)
            window_turns = turns[window_start:window_end]
            window_len = len(window_turns)
            is_final = window_end >= N

            self.windows_called += 1

            raw = ""
            parsed: list[dict[str, Any]] | None = None
            for _attempt in range(2):
                try:
                    raw = await self._call_llm(window_turns, known_projects)
                except Exception as e:
                    logger.warning(
                        "LLMSegmenterV3 LLM call failed (attempt %d): %s",
                        _attempt + 1, e,
                    )
                    raw = ""
                    continue
                parsed = self._parse_window(raw)
                if parsed is not None:
                    break
                self.json_parse_failures += 1

            if parsed is None:
                if is_final:
                    break
                window_start += self.stride
                continue

            any_window_succeeded = True
            all_windows_failed = False

            for raw_seg in parsed:
                v = self._validate_segment_dict(raw_seg, window_len)
                if v is None:
                    self.malformed_segments_dropped += 1
                    continue

                gstart = window_start + v["rel_start"]
                gend = window_start + v["rel_end"]

                if gend < frontier:
                    continue

                if gstart < frontier:
                    gstart = frontier
                    if gstart > gend:
                        continue

                rel_end_in_window = gend - window_start

                if (
                    not is_final
                    and rel_end_in_window >= window_len - self.defer_tail_turns
                ):
                    self.deferred_segments += 1
                    continue

                seg_turns = turns[gstart : gend + 1]
                emitted.append(
                    Segment(
                        turns=seg_turns,
                        topic_label=v["topic_label"],
                        start_idx=gstart,
                        end_idx=gend,
                        metadata={
                            "worthwhile": v["worthwhile"],
                            "score": v["score"],
                            "retrieval_key": v["retrieval_key"],
                            "summary": v["summary"],
                            "tags": v["tags"],
                            "outcome": v["outcome"],
                            "project": v["project"],
                            "source": "llm-segmenter-v3",
                            "window_idx": self.windows_called - 1,
                        },
                    )
                )
                frontier = gend + 1

            if is_final:
                break
            window_start += self.stride

        # If no window ever parsed successfully on a non-empty input, treat
        # as a hard parse failure so the caller can apply the fallback path.
        if not any_window_succeeded and N > 0:
            raise ValueError(
                f"LLMSegmenterV3: all {self.windows_called} windows failed to parse"
            )

        return emitted
