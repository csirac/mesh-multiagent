"""Live memory formation using the shared recall-oriented extraction contract.

The historical module/class names remain as compatibility aliases for callers,
but the behavior is extraction, not segmentation: records need not cover the
input, and one turn may produce multiple memories for distinct persistent
facts.  Prompt rendering and response validation live exclusively in
``mesh.memory.formation_contract``.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .entities import RegistryInjection
from .formation_contract import (
    FORMATION_PROMPT,
    FORMATION_CONTRACT_VERSION,
    FormationContractError,
    parse_formation_response,
    render_formation_prompt,
)

logger = logging.getLogger(__name__)

KNOWN_PROJECTS_CAP = 50
ENTITY_RESOLUTION_MODES = frozenset({"shadow", "write"})


@dataclass
class Segment:
    """Compatibility container for one extracted memory record.

    ``start_idx``/``end_idx`` identify the first evidence-bearing source turn;
    unlike the retired segmenter contract, records may share an index and do
    not collectively cover the input.
    """

    turns: list[Any]
    topic_label: str = ""
    start_idx: int = 0
    end_idx: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _timestamp_text(turn: Any) -> str:
    timestamp = getattr(turn, "timestamp", "") or ""
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    return str(timestamp)


def _canonical_turn_timestamp(turn: Any) -> str:
    """Return a retry-stable UTC timestamp with a ``Z`` suffix."""
    value = getattr(turn, "timestamp", turn)
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            parsed = datetime.fromisoformat(
                text[:-1] + "+00:00" if text.endswith("Z") else text
            )
        except ValueError:
            return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat().replace("+00:00", "Z")


def formation_window_key(
    cursor_start: int,
    cursor_end: int,
    first_turn: Any,
    last_turn: Any,
) -> str:
    """Derive the canonical durable key for one formation window."""
    serialized = "\x1f".join((
        str(cursor_start),
        str(cursor_end),
        _canonical_turn_timestamp(first_turn),
        _canonical_turn_timestamp(last_turn),
    ))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _render_turn(turn: Any, index: int) -> str:
    role = str(getattr(turn, "role", "") or "message")
    from_node = str(getattr(turn, "from_node", "") or "unknown")
    to_node = str(getattr(turn, "to_node", "") or "unknown")
    content = getattr(turn, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return (
        f"[turn:{index} | {_timestamp_text(turn)} | "
        f"{from_node} -> {to_node} | {role}]\n{content}"
    )


def _evidence_turn_index(trace: str, turns: list[Any]) -> int:
    """Locate the first source turn supporting a model-selected trace.

    The trace contract requires verbatim evidence.  Matching it back to source
    turns lets overlapping windows assign every record to exactly one window
    without adding source-index fields to the surviving field contract.
    """
    fragments = [
        line.strip()
        for line in trace.splitlines()
        if len(line.strip()) >= 8
    ]
    fragments.sort(key=len, reverse=True)
    for index, turn in enumerate(turns):
        content = getattr(turn, "content", "") or ""
        content = content if isinstance(content, str) else str(content)
        rendered = _render_turn(turn, index)
        if any(fragment in content or fragment in rendered for fragment in fragments):
            return index
    return 0


def _record_fingerprint(record: dict[str, Any]) -> tuple[str, ...]:
    return (
        record["event_date"],
        record["summary"].casefold(),
        record["retrieval_key"].casefold(),
        record["trace"],
    )


class LLMExtractorV1:
    """Windowed live extractor backed by the fold's lowbar contract."""

    # Compatibility surface for diagnostics/tests. This is the canonical file
    # content imported by formation_contract, not a second prompt.
    PROMPT = FORMATION_PROMPT

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
        agent_label: str = "the current mesh agent",
        digest_budget_tokens: int = 32_000,
        entity_resolution_mode: str = "off",
        entity_registry: RegistryInjection | None = None,
        entity_formation_max_tokens: int = 48_000,
    ) -> None:
        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must be in [0, window_size)")
        if defer_tail_turns < 0:
            raise ValueError("defer_tail_turns must be >= 0")
        if defer_tail_turns >= window_size:
            raise ValueError("defer_tail_turns must be < window_size")
        mode = str(entity_resolution_mode or "off").strip().lower()
        if mode not in {"off", *ENTITY_RESOLUTION_MODES}:
            raise ValueError(
                "entity_resolution_mode must be one of: off, shadow, write"
            )
        if entity_formation_max_tokens < 1:
            raise ValueError("entity_formation_max_tokens must be at least 1")
        self._llm_client = llm_client
        self.window_size = window_size
        self.overlap = overlap
        # Retained for config/API compatibility. Extraction ownership is based
        # on the overlap stride because records no longer have segment ranges.
        self.defer_tail_turns = defer_tail_turns
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = model
        self.request_timeout = request_timeout
        self.agent_label = agent_label
        self.digest_budget_tokens = digest_budget_tokens
        self.entity_resolution_mode = mode
        self.entity_registry = entity_registry
        self.entity_formation_max_tokens = entity_formation_max_tokens
        self.windows_called = 0
        self.json_parse_failures = 0
        self.malformed_segments_dropped = 0
        self.deferred_segments = 0
        self.duplicate_records_dropped = 0
        self.window_telemetry: list[dict[str, Any]] = []
        self._last_call_metrics: dict[str, int] = {}
        self._last_parse_error: str = ""

    @property
    def stride(self) -> int:
        return self.window_size - self.overlap

    @property
    def request_max_tokens(self) -> int:
        """Output ceiling for one formation request.

        Entity mode adds an ``entity`` object to every record, so it needs
        *more* output than core formation, never less.  The ceiling is
        therefore floored at ``max_tokens``: a small
        ``entity_formation_max_tokens`` can widen the budget but can never
        starve it below what the same window would get with entities off.

        This floor is load-bearing on reasoning backends.  ``max_tokens``
        bounds reasoning tokens *plus* content, so an undersized ceiling is
        spent entirely on reasoning and the API returns an empty completion
        with ``finish_reason="length"`` — which then fails contract parsing
        for every retry.  See docs/plans/entity-resolution-increment-3.md.
        """
        if self.entity_resolution_mode not in ENTITY_RESOLUTION_MODES:
            return self.max_tokens
        return max(self.max_tokens, self.entity_formation_max_tokens)

    @staticmethod
    def _format_known_projects(
        known_projects: list[str] | str | None,
    ) -> str:
        """Render a bounded project hint and log shown/elided telemetry."""
        if isinstance(known_projects, str):
            raw_names = [
                name.strip()
                for name in known_projects.split(",")
                if name.strip()
            ]
        elif known_projects is None:
            raw_names = []
        else:
            raw_names = [
                str(name).strip()
                for name in known_projects
                if str(name).strip()
            ]

        names = list(dict.fromkeys(raw_names))
        shown = names[:KNOWN_PROJECTS_CAP]
        elided = max(0, len(names) - len(shown))
        logger.info(
            "memory formation known_projects: shown=%d elided=%d total=%d",
            len(shown),
            elided,
            len(names),
        )
        if not shown:
            return (
                "### KNOWN PROJECTS\n"
                "No known projects. Use a concise lowercase project slug, "
                '"personal", or an empty string as the canonical contract allows.'
            )
        return (
            "### KNOWN PROJECTS\n"
            + "\n".join(f"- {name}" for name in shown)
            + (
                f"\n- ({elided} additional project names elided)"
                if elided
                else ""
            )
            + "\nUse an exact listed slug when one applies; otherwise use a "
            'concise new slug, "personal", or an empty string.'
        )

    async def _call_llm(
        self,
        window_turns: list[Any],
        known_projects: list[str] | str | None = None,
    ) -> str:
        prompt = render_formation_prompt(
            agent_label=self.agent_label,
            budget_tokens=self.digest_budget_tokens,
            entity_resolution_mode=self.entity_resolution_mode,
        )
        rendered_turns = "\n\n".join(
            _render_turn(turn, index)
            for index, turn in enumerate(window_turns)
        )
        entity_block = ""
        if self.entity_resolution_mode in ENTITY_RESOLUTION_MODES:
            entity_block = (
                self.entity_registry.payload
                if self.entity_registry is not None
                else (
                    "### ENTITY REGISTRY\n"
                    "No active or pending entities."
                )
            )
            entity_block += "\n\n"
        request = (
            f"{prompt}\n\n"
            f"{self._format_known_projects(known_projects)}\n\n"
            f"{entity_block}"
            "### RAW HISTORY WINDOW\n"
            f"{rendered_turns}\n\n"
            "Emit the JSON array of memory records now."
        )
        from ..llm import estimate_tokens

        started = time.perf_counter()
        raw = ""
        try:
            raw = await asyncio.wait_for(
                self._llm_client.complete(
                    request,
                    model=self.model,
                    max_tokens=self.request_max_tokens,
                    temperature=self.temperature,
                ),
                timeout=self.request_timeout,
            )
            return raw
        finally:
            usage = getattr(self._llm_client, "_last_usage", None)
            usage = usage if isinstance(usage, dict) else {}
            self._last_call_metrics = {
                "input_tokens": int(
                    usage.get("input_tokens") or estimate_tokens(request)
                ),
                "output_tokens": int(
                    usage.get("output_tokens") or estimate_tokens(raw)
                ),
                "latency_ms": max(
                    0, int((time.perf_counter() - started) * 1000)
                ),
            }

    def _parse_window(self, raw: str) -> list[dict[str, Any]] | None:
        """Compatibility wrapper over the single shared parser."""
        try:
            self._last_parse_error = ""
            return parse_formation_response(
                raw,
                entity_resolution_mode=self.entity_resolution_mode,
                known_entity_statuses=(
                    self.entity_registry.statuses
                    if self.entity_registry is not None
                    else {}
                ),
                allowed_entity_keys=(
                    frozenset(self.entity_registry.entity_keys)
                    if self.entity_registry is not None
                    else frozenset()
                ),
            )
        except FormationContractError as exc:
            # Keep the concrete reason: "formation response is empty" (the
            # reasoning-budget starvation signature) and a truncated-JSON
            # decode error are different faults with different fixes, and the
            # generic telemetry string hid that distinction.
            self._last_parse_error = str(exc)
            return None

    async def segment(
        self,
        turns: list[Any],
        *,
        known_projects: list[str] | str | None = None,
        cursor_start: int = 0,
    ) -> list[Segment]:
        """Extract recall-oriented records from overlapping input windows."""
        if not turns:
            return []

        emitted: list[Segment] = []
        seen: set[tuple[str, ...]] = set()
        total_turns = len(turns)
        window_start = 0
        any_window_succeeded = False

        while window_start < total_turns:
            window_end = min(window_start + self.window_size, total_turns)
            window_turns = turns[window_start:window_end]
            is_final = window_end >= total_turns
            self.windows_called += 1
            absolute_start = cursor_start + window_start
            absolute_end = cursor_start + window_end
            durable_window_key = formation_window_key(
                absolute_start,
                absolute_end,
                window_turns[0],
                window_turns[-1],
            )
            telemetry = {
                "window_key": durable_window_key,
                "agent": self.agent_label,
                "mode": self.entity_resolution_mode,
                "contract_version": FORMATION_CONTRACT_VERSION,
                "window_start": absolute_start,
                "window_end": absolute_end,
                "candidates_injected": (
                    self.entity_registry.candidates_injected
                    if self.entity_registry is not None
                    else 0
                ),
                "existing_links_made": 0,
                "new_proposals": 0,
                "unresolved_count": 0,
                "validation_failures": 0,
                "validation_failure_reasons": [],
                "serialized_registry_token_count": (
                    self.entity_registry.serialized_token_count
                    if self.entity_registry is not None
                    else 0
                ),
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0,
            }

            parsed: list[dict[str, Any]] | None = None
            for attempt in range(2):
                try:
                    raw = await self._call_llm(window_turns, known_projects)
                except Exception as exc:
                    for key in ("input_tokens", "output_tokens", "latency_ms"):
                        telemetry[key] += self._last_call_metrics.get(key, 0)
                    logger.warning(
                        "LLMExtractorV1 call failed (attempt %d): %s",
                        attempt + 1,
                        exc,
                    )
                    continue
                for key in ("input_tokens", "output_tokens", "latency_ms"):
                    telemetry[key] += self._last_call_metrics.get(key, 0)
                parsed = self._parse_window(raw)
                if parsed is not None:
                    break
                self.json_parse_failures += 1

            if parsed is None:
                if self.entity_resolution_mode in ENTITY_RESOLUTION_MODES:
                    telemetry["validation_failures"] += 1
                    reason = (
                        "formation response failed contract parsing after "
                        "2 attempts"
                    )
                    if self._last_parse_error:
                        reason = f"{reason}: {self._last_parse_error}"
                    telemetry["validation_failure_reasons"].append(reason)
                    telemetry["created_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    self.window_telemetry.append(telemetry)
                raise ValueError(
                    "LLMExtractorV1: window "
                    f"{self.windows_called - 1} failed after 2 attempts"
                )

            any_window_succeeded = True
            if self.entity_resolution_mode in ENTITY_RESOLUTION_MODES:
                for record in parsed:
                    reasons = record.get("entity_validation_failures") or []
                    telemetry["validation_failures"] += len(reasons)
                    telemetry["validation_failure_reasons"].extend(reasons)
                telemetry["created_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                self.window_telemetry.append(telemetry)
            owner_end = total_turns if is_final else window_start + self.stride
            for record in parsed:
                source_rel = _evidence_turn_index(record["trace"], window_turns)
                source_global = window_start + source_rel
                if source_global >= owner_end:
                    self.deferred_segments += 1
                    continue
                fingerprint = _record_fingerprint(record)
                if fingerprint in seen:
                    self.duplicate_records_dropped += 1
                    continue
                seen.add(fingerprint)
                if self.entity_resolution_mode in ENTITY_RESOLUTION_MODES:
                    entity = record.get("entity") or {}
                    telemetry["existing_links_made"] += len(
                        entity.get("existing_keys") or []
                    )
                    telemetry["new_proposals"] += len(
                        entity.get("new_entities") or []
                    )
                    telemetry["unresolved_count"] += len(
                        entity.get("unresolved") or []
                    )
                emitted.append(
                    Segment(
                        turns=[turns[source_global]],
                        topic_label=record["topic_label"] or "untitled",
                        start_idx=source_global,
                        end_idx=source_global,
                        metadata={
                            **record,
                            "source": "live-extraction",
                            "window_idx": self.windows_called - 1,
                            "window_key": durable_window_key,
                            "window_start": absolute_start,
                            "window_end": absolute_end,
                        },
                    )
                )

            if is_final:
                break
            window_start += self.stride

        if not any_window_succeeded:
            raise ValueError(
                f"LLMExtractorV1: all {self.windows_called} windows failed to parse"
            )
        return emitted


# Backward-compatible import name. There is only one implementation.
LLMSegmenterV3 = LLMExtractorV1
