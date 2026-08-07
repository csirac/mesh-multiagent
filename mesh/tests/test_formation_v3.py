"""Tests for Memory Formation v3 — segmenter + integration.

Plan: docs/plans/memory-formation-v3-2026-04-27.md (rev 6, §5.1).

Covers:
- Tests 1-8: LLMSegmenterV3 unit tests (parse/validate/window-stepping).
- Tests 9-14: Integration with mocked LLM through MemorySystemV2.
"""
from __future__ import annotations

import asyncio
import json
import time
import tempfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from mesh.conversation_history import Turn
from mesh.memory.entities import RegistryInjection
from mesh.memory.formation_v3 import LLMSegmenterV3, formation_window_key
from mesh.memory.store import MemoryEntry, MemoryStore
from mesh.memory.system_v2 import MemorySystemV2


# ── Helpers ─────────────────────────────────────────────────────────


def _turn(i: int, content: str = "", role: str = "user", from_node: str = "user:testuser") -> Turn:
    return Turn(
        role=role,
        content=content or f"turn-{i} content",
        timestamp=datetime(2026, 4, 27, 10, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=i),
        from_node=from_node,
        seq_id=i,
    )


def _mk_segments_json(specs: list[dict]) -> str:
    """Build a canonical extraction-contract response."""
    return json.dumps(specs)


def _mk_seg(start: int, end: int, **overrides) -> dict:
    base = {
        "topic_label": f"topic-{start}-{end}",
        "retrieval_key": f"key for segment {start}-{end}",
        "summary": f"summary {start}-{end}",
        "reflection": f"reflection {start}-{end}",
        "trace": f"turn-{start} content",
        "tags": ["test"],
        "outcome": "success",
        "project": "",
        "event_date": "2026-04-27",
        "digest_candidate": True,
    }
    base.update(overrides)
    return base


class _FakeLLMClient:
    """Mock LLM client supporting `complete()` with a configurable response."""

    def __init__(self, responses: list[str] | None = None, side_effect=None):
        self._responses = list(responses) if responses else []
        self.side_effect = side_effect
        self.calls: list[dict] = []
        self.call_timestamps: list[float] = []

    async def complete(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        self.call_timestamps.append(time.time())
        if self.side_effect:
            res = self.side_effect(prompt, kwargs)
            if asyncio.iscoroutine(res):
                return await res
            return res
        if not self._responses:
            return ""
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


class _FakeEmbedder:
    """Mock embedder that records inputs and returns deterministic vectors."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.batch_inputs: list[list[str]] = []
        self.single_inputs: list[str] = []

    async def embed_batch_to_arrays(self, texts: list[str]) -> list[np.ndarray]:
        self.batch_inputs.append(list(texts))
        out = []
        for t in texts:
            rng = np.random.RandomState(abs(hash(t)) % 2**31)
            out.append(rng.randn(self.dim).astype(np.float32))
        return out

    async def embed_to_array(self, text: str) -> np.ndarray:
        self.single_inputs.append(text)
        rng = np.random.RandomState(abs(hash(text)) % 2**31)
        return rng.randn(self.dim).astype(np.float32)

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_to_array(text)).tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        arrs = await self.embed_batch_to_arrays(texts)
        return [a.tolist() for a in arrs]


# ── Unit: parse / validate ──────────────────────────────────────────


class TestParseValidate:
    def _seg(self, llm=None):
        return LLMSegmenterV3(
            llm or _FakeLLMClient(), window_size=10, overlap=3, defer_tail_turns=2,
        )

    def test_parse_window_valid_json(self):
        """Test 1: canonical valid JSON → list of dicts."""
        s = self._seg()
        raw = _mk_segments_json([_mk_seg(0, 4)])
        parsed = s._parse_window(raw)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["summary"] == "summary 0-4"
        assert parsed[0]["trace"] == "turn-0 content"
        assert parsed[0]["digest_candidate"] is True

    def test_parse_window_handles_code_fences(self):
        """Test 2: ```json {...} ``` is stripped."""
        s = self._seg()
        raw = "```json\n" + _mk_segments_json([_mk_seg(0, 3)]) + "\n```"
        parsed = s._parse_window(raw)
        assert isinstance(parsed, list)
        assert parsed[0]["event_date"] == "2026-04-27"

    def test_parse_window_returns_none_on_garbage(self):
        """Test 3: non-JSON returns None."""
        s = self._seg()
        assert s._parse_window("not json at all { broken") is None
        assert s._parse_window("[]") == []
        assert s._parse_window("{}") is None
        assert s._parse_window('{"segments": "not a list"}') is None

    def test_parser_requires_full_contract(self):
        """Test 4: every surviving fold field is required."""
        s = self._seg()
        missing_trace = _mk_seg(0, 3)
        del missing_trace["trace"]
        assert s._parse_window(json.dumps([missing_trace])) is None

    def test_parser_normalises_outcome_and_tags(self):
        """Test 5: outcome casing and comma-separated tags are normalized."""
        s = self._seg()
        parsed = s._parse_window(_mk_segments_json([
            _mk_seg(0, 3, outcome="Success", tags="one,two"),
        ]))
        assert parsed[0]["outcome"] == "success"
        assert parsed[0]["tags"] == ["one", "two"]
        assert s._parse_window(_mk_segments_json([
            _mk_seg(0, 3, outcome="winning"),
        ])) is None

    def test_parser_rejects_bad_event_date_and_digest_candidate(self):
        """Test 6: date and two-bar tag have strict types."""
        s = self._seg()
        assert s._parse_window(_mk_segments_json([
            _mk_seg(0, 3, event_date="April 27"),
        ])) is None
        assert s._parse_window(_mk_segments_json([
            _mk_seg(0, 3, digest_candidate="false"),
        ])) is None


class TestWindowStepping:
    @pytest.mark.asyncio
    async def test_overlap_defer_emits_correct_ranges(self):
        """Test 7: evidence ownership prevents overlap double-emission."""
        # window_size=20, overlap=5, defer_tail=3 → stride=15
        # 200 turns → windows at [0,20), [15,35), [30,50), ...
        turns = [_turn(i, f"turn-content-{i}") for i in range(200)]

        def fake_response(prompt: str, kwargs: dict) -> str:
            # Simpler: extract last_idx from prompt — not stable. Use call counter.
            window_idx = fake_response.calls
            fake_response.calls += 1
            window_len = 20 if window_idx < (200 // 15) else (200 - window_idx * 15)
            window_len = max(1, min(window_len, 20))
            global_start = window_idx * 15
            last_global = global_start + window_len - 1
            segs = [
                _mk_seg(
                    0,
                    0,
                    summary=f"window {window_idx} first",
                    retrieval_key=f"window {window_idx} first",
                    trace=f"turn-content-{global_start}",
                ),
                _mk_seg(
                    window_len - 1,
                    window_len - 1,
                    summary=f"window {window_idx} tail",
                    retrieval_key=f"window {window_idx} tail",
                    trace=f"turn-content-{last_global}",
                ),
            ]
            return _mk_segments_json(segs)
        fake_response.calls = 0

        llm = _FakeLLMClient(side_effect=fake_response)
        seg = LLMSegmenterV3(
            llm, window_size=20, overlap=5, defer_tail_turns=3,
        )
        emitted = await seg.segment(turns)

        assert len({record.start_idx for record in emitted}) == len(emitted)
        assert [record.start_idx for record in emitted] == sorted(
            record.start_idx for record in emitted
        )
        assert seg.deferred_segments > 0, "expected some segments deferred"
        assert emitted[-1].end_idx == 199

    @pytest.mark.asyncio
    async def test_digest_candidate_does_not_filter_searchable_memories(self):
        """Test 8: the lowbar forms both digest and DB-only records."""
        turns = [_turn(i) for i in range(30)]
        llm = _FakeLLMClient(responses=[
            _mk_segments_json([
                _mk_seg(0, 9, summary="digest event", digest_candidate=True),
                _mk_seg(
                    10,
                    19,
                    summary="search-only aside",
                    digest_candidate=False,
                ),
            ])
        ])
        seg = LLMSegmenterV3(
            llm, window_size=30, overlap=5, defer_tail_turns=3,
        )
        emitted = await seg.segment(turns)
        assert len(emitted) == 2
        assert [record.metadata["digest_candidate"] for record in emitted] == [
            True,
            False,
        ]

    @pytest.mark.asyncio
    async def test_entity_registry_injection_limit_and_durable_window_key(self):
        response = _mk_segments_json([
            _mk_seg(
                0,
                0,
                entity={
                    "existing_keys": ["person:owner"],
                    "new_entities": [],
                    "unresolved": [],
                },
            )
        ])
        registry = RegistryInjection(
            payload=(
                "### ENTITY REGISTRY\n"
                '{"key":"person:owner","type":"person","display_name":"Project Owner",'
                '"aliases":"owner","identity_note":"Primary user"}'
            ),
            entity_keys=("person:owner",),
            statuses={"person:owner": "active"},
            candidates_injected=1,
            serialized_token_count=31,
        )
        first_llm = _FakeLLMClient(responses=[response])
        first = LLMSegmenterV3(
            first_llm,
            window_size=10,
            overlap=2,
            defer_tail_turns=1,
            entity_resolution_mode="shadow",
            entity_registry=registry,
            entity_formation_max_tokens=1200,
        )
        turns = [_turn(0)]
        first_records = await first.segment(turns, cursor_start=37)

        second_llm = _FakeLLMClient(responses=[response])
        second = LLMSegmenterV3(
            second_llm,
            window_size=10,
            overlap=2,
            defer_tail_turns=1,
            entity_resolution_mode="shadow",
            entity_registry=registry,
            entity_formation_max_tokens=1200,
        )
        second_records = await second.segment(turns, cursor_start=37)

        expected = formation_window_key(37, 38, turns[0], turns[0])
        assert first_records[0].metadata["window_key"] == expected
        assert second_records[0].metadata["window_key"] == expected
        assert "### ENTITY REGISTRY" in first_llm.calls[0]["prompt"]
        # Entity mode is floored at the core formation budget: a small
        # entity ceiling may widen it, never starve it below core formation.
        assert first_llm.calls[0]["max_tokens"] == first.max_tokens
        assert first.window_telemetry[0]["serialized_registry_token_count"] == 31

        changed = formation_window_key(38, 39, turns[0], turns[0])
        assert changed != expected

    @pytest.mark.asyncio
    async def test_window_key_is_stable_across_contract_retry(self):
        response = _mk_segments_json([
            _mk_seg(
                0,
                0,
                entity={
                    "existing_keys": [],
                    "new_entities": [],
                    "unresolved": [],
                },
            )
        ])
        registry = RegistryInjection(
            payload="### ENTITY REGISTRY\nNo active or pending entities.",
            entity_keys=(),
            statuses={},
            candidates_injected=0,
            serialized_token_count=12,
        )
        llm = _FakeLLMClient(responses=["not-json", response])
        extractor = LLMSegmenterV3(
            llm,
            window_size=10,
            overlap=2,
            defer_tail_turns=1,
            entity_resolution_mode="shadow",
            entity_registry=registry,
        )
        turns = [_turn(0)]

        records = await extractor.segment(turns, cursor_start=12)

        expected = formation_window_key(12, 13, turns[0], turns[0])
        assert records[0].metadata["window_key"] == expected
        assert extractor.window_telemetry[0]["window_key"] == expected
        assert len(llm.calls) == 2


class TestEntityFormationOutputBudget:
    """Entity mode must never get a smaller output ceiling than core formation.

    Regression: entity mode passed ``entity_formation_max_tokens`` (1,200) as
    the request ceiling in place of the 8,000 used with entities off. On a
    reasoning backend that ceiling bounds reasoning tokens plus content, so it
    was spent entirely on reasoning; the API returned an empty completion and
    earlier formation runs failed contract parsing twice over.
    """

    def _segmenter(self, mode: str, entity_max: int, **kwargs):
        return LLMSegmenterV3(
            _FakeLLMClient(),
            window_size=10,
            overlap=2,
            defer_tail_turns=1,
            entity_resolution_mode=mode,
            entity_formation_max_tokens=entity_max,
            **kwargs,
        )

    def test_entity_ceiling_is_floored_at_core_budget(self):
        seg = self._segmenter("write", 1200)
        assert seg.max_tokens == 8000
        assert seg.request_max_tokens == 8000

    def test_entity_ceiling_can_widen_beyond_core_budget(self):
        seg = self._segmenter("shadow", 48_000)
        assert seg.request_max_tokens == 48_000

    def test_off_mode_keeps_core_budget(self):
        seg = self._segmenter("off", 1200)
        assert seg.request_max_tokens == 8000

    @pytest.mark.asyncio
    async def test_request_uses_floored_ceiling(self):
        llm = _FakeLLMClient(responses=[
            _mk_segments_json([
                _mk_seg(
                    0,
                    0,
                    entity={
                        "existing_keys": [],
                        "new_entities": [],
                        "unresolved": [],
                    },
                )
            ])
        ])
        seg = LLMSegmenterV3(
            llm,
            window_size=10,
            overlap=2,
            defer_tail_turns=1,
            entity_resolution_mode="write",
            entity_formation_max_tokens=1200,
        )
        await seg.segment([_turn(0)])
        assert llm.calls[0]["max_tokens"] == 8000

    @pytest.mark.asyncio
    async def test_empty_completion_reason_reaches_telemetry(self):
        """An empty completion is the reasoning-starvation signature.

        The generic "failed contract parsing" string hid which fault occurred;
        the concrete parser message must survive into the JSONL telemetry.
        """
        llm = _FakeLLMClient(responses=[""])
        seg = LLMSegmenterV3(
            llm,
            window_size=10,
            overlap=2,
            defer_tail_turns=1,
            entity_resolution_mode="write",
        )
        with pytest.raises(ValueError):
            await seg.segment([_turn(0)])
        reasons = seg.window_telemetry[0]["validation_failure_reasons"]
        assert len(reasons) == 1
        assert "failed contract parsing after 2 attempts" in reasons[0]
        assert "formation response is empty" in reasons[0]


class TestCorrectionSummaryGuidance:
    @pytest.mark.asyncio
    async def test_correction_summary_abstracts_retracted_claim(self):
        """Correction episodes retain the remediation, not the false assertion.

        The false assertion is deliberately present in the source turns. The
        shared contract's evidence and quote discipline keeps the correction
        grounded without mechanically copying raw turns into the summary.
        """
        false_claim = "Kaylee has a son named Charlie"
        turns = [
            _turn(0, false_claim, role="assistant", from_node="agent:assistant:alice"),
            _turn(
                1,
                "That detail was fabricated test data. Please remove it from Kaylee's essay.",
            ),
            _turn(
                2,
                "I will add an essay-editing tool and correct the essay.",
                role="assistant",
                from_node="agent:sysadmin:bob",
            ),
        ]

        def correction_response(prompt: str, _kwargs: dict) -> str:
            assert "Quote discipline" in prompt
            assert "trace field must prove it" in prompt
            return _mk_segments_json([
                _mk_seg(
                    0,
                    2,
                    topic_label="Kaylee essay correction",
                    retrieval_key="Correct inaccurate Kaylee biography in essay memory",
                    summary=(
                        "Alice served inaccurate biographical details about Kaylee based on "
                        "fabricated test data. Project Owner identified the error, and Bob added an "
                        "essay-editing tool so it could be corrected."
                    ),
                    tags=["memory-system", "correction"],
                    trace=(
                        "That detail was fabricated test data. "
                        "Please remove it from Kaylee's essay."
                    ),
                )
            ])

        llm = _FakeLLMClient(side_effect=correction_response)
        segmenter = LLMSegmenterV3(
            llm, window_size=10, overlap=2, defer_tail_turns=1,
        )

        segments = await segmenter.segment(turns)

        assert len(segments) == 1
        metadata = segments[0].metadata
        summary = metadata["summary"]
        assert false_claim not in summary
        assert "Charlie" not in summary
        assert "son" not in summary.lower()
        assert "inaccurate biographical details about Kaylee" in summary
        assert "essay-editing tool" in summary
        assert "Charlie" not in segments[0].topic_label
        assert "son" not in segments[0].topic_label.lower()
        assert "Charlie" not in metadata["retrieval_key"]
        assert "son" not in metadata["retrieval_key"].lower()
        assert all("Charlie" not in tag and "son" not in tag.lower() for tag in metadata["tags"])


# ── Integration: mocked LLM through MemorySystemV2 ──────────────────


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _build_memory_system(tmp_dir, llm, embedder, **kwargs):
    sys = MemorySystemV2(
        nickname=kwargs.pop("nickname", "test-agent"),
        llm_client=llm,
        formation_v3_enabled=kwargs.pop("formation_v3_enabled", True),
        formation_v3_window_size=kwargs.pop("formation_v3_window_size", 20),
        formation_v3_overlap=kwargs.pop("formation_v3_overlap", 5),
        formation_v3_defer_tail=kwargs.pop("formation_v3_defer_tail", 3),
        **kwargs,
    )
    sys._store = MemoryStore(kwargs.get("nickname", "test-agent"), db_dir=tmp_dir)
    sys._embedder = embedder
    sys._formation_lock = asyncio.Lock()
    return sys


class TestIntegration:
    @pytest.mark.asyncio
    async def test_formation_creates_integration_task(self, tmp_dir):
        """Test 9: after formation, entries trigger a fire-and-forget integration task."""
        llm = _FakeLLMClient(responses=[
            _mk_segments_json([_mk_seg(0, 4, worthwhile=True, project="test-proj")])
        ])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        integrate_calls = []

        async def fake_integrate(entries):
            integrate_calls.append(entries)

        sys._integrate_entries_into_maps = fake_integrate

        turns = [_turn(i) for i in range(5)]
        n = await sys.form_un_formed(turns, "time-based")
        assert n == 1
        # Let the fire-and-forget task run.
        await asyncio.sleep(0.05)
        assert len(integrate_calls) == 1
        assert len(integrate_calls[0]) == 1
        sys._store.close()

    @pytest.mark.asyncio
    async def test_integration_failure_does_not_lose_memories(self, tmp_dir):
        """Test 10: integration raises; entries still persisted; cursor advances."""
        llm = _FakeLLMClient(responses=[
            _mk_segments_json([_mk_seg(0, 4, worthwhile=True, summary="keep")])
        ])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        async def fail_integrate(entries):
            raise RuntimeError("integration failed")

        sys._integrate_entries_into_maps = fail_integrate
        turns = [_turn(i) for i in range(5)]
        n = await sys.form_un_formed(turns, "time-based")
        assert n == 1
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 5
        assert len(sys._pool) == 1
        # Let the fire-and-forget task finish (and fail silently).
        await asyncio.sleep(0.05)
        sys._store.close()

    @pytest.mark.asyncio
    async def test_segmenter_failure_does_not_advance_cursor(self, tmp_dir):
        """Test 11: segmenter raises; cursor NOT advanced; no integration task."""
        async def fail_seg(prompt, kwargs):
            raise RuntimeError("segmenter failed")

        llm = _FakeLLMClient(side_effect=fail_seg)
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        integrate_calls = []

        async def fake_integrate(entries):
            integrate_calls.append(entries)

        sys._integrate_entries_into_maps = fake_integrate
        turns = [_turn(i) for i in range(5)]
        n = await sys.form_un_formed(turns, "time-based")
        assert n == 0
        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 0  # NOT advanced
        await asyncio.sleep(0.05)
        assert len(integrate_calls) == 0  # No entries → no integration
        sys._store.close()

    @pytest.mark.asyncio
    async def test_v3_flag_disabled_uses_legacy_path(self, tmp_dir):
        """Test 12: with v3 disabled, on_window_drop runs legacy `_segment_by_topic`."""
        llm = _FakeLLMClient()
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder, formation_v3_enabled=False)

        # Track legacy calls.
        seg_called = []
        sys._segment_by_topic = lambda turns: (seg_called.append(turns) or [])

        # _update_conversation_summary should also be called by legacy path.
        usc_called = []

        async def usc(turns, count):
            usc_called.append(count)

        sys._update_conversation_summary = usc
        turns = [_turn(i) for i in range(10)]
        await sys.on_window_drop(turns)
        assert len(seg_called) == 1
        assert len(usc_called) == 1

        # form_un_formed should be a no-op when v3 disabled.
        n = await sys.form_un_formed(turns, "time-based")
        assert n == 0
        sys._store.close()

    @pytest.mark.asyncio
    async def test_persist_v3_entry_field_mapping(self, tmp_dir):
        """Test 14: verify field mapping for persisted MemoryEntry.

        Asserts:
        - reflection_embedding computed on `summary + " " + retrieval_key`
        - reflection and trace are model-selected fields
        - digest_candidate and event_date survive persistence
        - provenance identifies authoritative live extraction
        """
        llm = _FakeLLMClient(responses=[
            _mk_segments_json([
                _mk_seg(0, 4,
                        summary="A great summary",
                        reflection="Why this mattered",
                        trace="turn-0 content",
                        retrieval_key="key for retrieval",
                        outcome="", project="mesh-tools",
                        topic_label="my topic",
                        tags=["x", "y"],
                        digest_candidate=False),
            ])
        ])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)
        turns = [_turn(i) for i in range(5)]
        await sys.form_un_formed(turns, "time-based")

        assert len(sys._pool) == 1
        e = sys._pool[0]
        assert e.summary == "A great summary"
        assert e.reflection == "Why this mattered"
        assert e.trace == "turn-0 content"
        assert e.retrieval_key == "key for retrieval"
        assert e.topic_label == "my topic"
        assert "x" in e.tags and "y" in e.tags
        assert not any(tag.startswith("score:") for tag in e.tags)
        assert e.outcome == ""
        assert e.project == "mesh-tools"
        assert e.event_date == "2026-04-27"
        assert e.digest_candidate is False
        assert e.formation_source == "live-extraction"
        # Embedder was called with summary + " " + retrieval_key.
        # First batch is reflection_embedding targets.
        assert embedder.batch_inputs[0] == ["A great summary key for retrieval"]
        # Second batch is retrieval_key embeddings.
        assert embedder.batch_inputs[1] == ["key for retrieval"]
        sys._store.close()


# ── Project-name hallucination fixes (canary follow-up) ─────────────


class TestKnownProjectsConstraint:
    """Fix (a): the v3 segmenter prompt must include the agent's known
    project names so the LLM picks from the list rather than inventing."""

    @pytest.mark.asyncio
    async def test_v3_segmenter_receives_known_projects_in_prompt(self):
        """When known_projects is passed, the rendered prompt contains
        the project names under the bounded known-projects section."""
        llm = _FakeLLMClient(responses=[
            _mk_segments_json([_mk_seg(0, 4, worthwhile=True, project="sp26-221")])
        ])
        seg = LLMSegmenterV3(llm, window_size=10, overlap=2, defer_tail_turns=1)
        turns = [_turn(i) for i in range(5)]

        # Pass as a comma-separated string (the format
        # MemorySystemV2._known_project_names() returns).
        await seg.segment(turns, known_projects="sp26-221, study-pipeline, llm-eval")

        assert len(llm.calls) == 1
        prompt = llm.calls[0]["prompt"]
        assert "### KNOWN PROJECTS" in prompt
        assert "- sp26-221" in prompt
        assert "- study-pipeline" in prompt
        assert "- llm-eval" in prompt
        assert "Use an exact listed slug" in prompt

    @pytest.mark.asyncio
    async def test_v3_segmenter_handles_empty_known_projects(self):
        """When no known projects are provided, the prompt encourages
        the LLM to suggest project labels (cold-start behavior)."""
        llm = _FakeLLMClient(responses=[
            _mk_segments_json([_mk_seg(0, 4, worthwhile=True, project=None)])
        ])
        seg = LLMSegmenterV3(llm, window_size=10, overlap=2, defer_tail_turns=1)
        turns = [_turn(i) for i in range(5)]

        # None
        await seg.segment(turns, known_projects=None)
        assert "No known projects." in llm.calls[0]["prompt"]

        # Empty string
        llm.calls.clear()
        llm._responses = [_mk_segments_json([_mk_seg(0, 4, worthwhile=True)])]
        await seg.segment(turns, known_projects="")
        assert "No known projects." in llm.calls[0]["prompt"]

        # Empty list
        llm.calls.clear()
        llm._responses = [_mk_segments_json([_mk_seg(0, 4, worthwhile=True)])]
        await seg.segment(turns, known_projects=[])
        assert "No known projects." in llm.calls[0]["prompt"]

    @pytest.mark.asyncio
    async def test_v3_segmenter_accepts_list_of_strings(self):
        """list[str] is rendered identically to a comma-separated string."""
        llm = _FakeLLMClient(responses=[
            _mk_segments_json([_mk_seg(0, 4, worthwhile=True)])
        ])
        seg = LLMSegmenterV3(llm, window_size=10, overlap=2, defer_tail_turns=1)
        turns = [_turn(i) for i in range(5)]
        await seg.segment(turns, known_projects=["sp26-221", "study-pipeline"])
        prompt = llm.calls[0]["prompt"]
        assert "- sp26-221" in prompt
        assert "- study-pipeline" in prompt


class TestCreateNewProjectMapShortCircuit:
    """Fix (c): _create_new_project_map must skip the LLM call when no
    project_dir is resolvable, since the eventual create_map() will fail
    anyway. Saves ~4096 tokens + 60s timeout per hallucinated name."""

    @pytest.mark.asyncio
    async def test_create_new_project_map_always_proceeds_without_project_dir(
        self, tmp_dir
    ):
        """No active_project_dir → bootstrap still proceeds (central dir)."""
        llm = _FakeLLMClient(responses=[
            "# Project: class-sp26\n\n## Summary\n\nA new project.\n\n## Goals\n\nTBD"
        ])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)
        sys._active_project_dir = None

        await sys._create_new_project_map(
            "class-sp26", "Some hallucinated summary"
        )

        assert len(llm.calls) == 1
        content = await sys.get_map("class-sp26")
        assert content is not None
        assert "class-sp26" in content
        sys._store.close()

    @pytest.mark.asyncio
    async def test_create_new_project_map_proceeds_when_project_dir_resolvable(
        self, tmp_dir
    ):
        """Active_project_dir set → LLM IS called (control case)."""
        llm = _FakeLLMClient(responses=[
            "# Project: real-project\n\nSome map content here."
        ])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)
        # Resolvable via _active_project_dir.
        sys._active_project_dir = tmp_dir

        await sys._create_new_project_map("real-project", "valid summary")

        assert len(llm.calls) == 1
        sys._store.close()


# ── Entry-Driven Map Integration ──────────────────────────────────────


def _mk_entry(project: str = "test-proj", **overrides) -> MemoryEntry:
    """Create a MemoryEntry for integration tests."""
    defaults = dict(
        id=MemoryEntry.new_id(),
        created_at=datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc),
        summary="Implemented feature X with approach Y.",
        reflection="",
        trace="",
        trigger="[TOPIC: feature-x] user asked about X",
        retrieval_key="implement feature X",
        topic_label="feature-x",
        project=project,
        tags=["impl", "score:7"],
        outcome="success",
    )
    defaults.update(overrides)
    return MemoryEntry(**defaults)


class TestEntryIntegration:
    """Tests for _integrate_entries_into_maps and supporting methods."""

    def test_format_entries_for_integration(self):
        """Entry formatting includes all expected fields."""
        entry = _mk_entry(
            topic_label="socket-fix",
            retrieval_key="fix predictable socket path",
            tags=["security", "score:8"],
            outcome="success",
            summary="Moved Unix socket to ~/.mesh/sockets/ with mode 0700.",
        )
        from mesh.memory.system_v2 import MemorySystemV2
        text = MemorySystemV2._format_entries_for_integration([entry])
        assert "fix predictable socket path" in text
        assert "socket-fix" in text
        assert "8/10" in text
        assert "success" in text
        assert "security" in text
        assert "score:" not in text.split("Tags:")[1].split("\n")[0]
        assert "Moved Unix socket" in text

    @pytest.mark.asyncio
    async def test_single_project_integration(self, tmp_dir):
        """Single project with entries → map updated via LLM."""
        llm = _FakeLLMClient(responses=[
            '<append_to name="## Open Issues">\n'
            "- **Socket security** — predictable path\n"
            "</append_to>"
        ])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        # Create initial map.
        initial_map = "# Project: test-proj\n\n## Open Issues\n- **Old issue** — stale\n"
        await sys.create_map("test-proj", initial_map, project_dir=tmp_dir)

        entry = _mk_entry(project="test-proj")
        await sys._integrate_entries_into_maps([entry])

        updated = await sys.get_map("test-proj")
        assert "Socket security" in updated
        assert "Old issue" in updated
        sys._store.close()

    @pytest.mark.asyncio
    async def test_multi_project_integration(self, tmp_dir):
        """Entries spanning two projects → each gets separate LLM call."""
        call_count = {"n": 0}

        async def counting_complete(prompt, **kwargs):
            call_count["n"] += 1
            return "No updates needed."

        llm = _FakeLLMClient(side_effect=lambda p, k: counting_complete(p, **k))
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        # Each project needs its own directory (maps are PROJECT_MAP.md).
        import os
        for proj in ("proj-a", "proj-b"):
            proj_dir = os.path.join(tmp_dir, proj)
            os.makedirs(proj_dir, exist_ok=True)
            content = f"# Project: {proj}\n\n## Current State\nActive.\n"
            await sys.create_map(proj, content, project_dir=proj_dir)

        entries = [
            _mk_entry(project="proj-a", summary="Work on A"),
            _mk_entry(project="proj-b", summary="Work on B"),
        ]
        await sys._integrate_entries_into_maps(entries)

        # Two projects → two LLM calls.
        assert call_count["n"] == 2
        sys._store.close()

    @pytest.mark.asyncio
    async def test_no_map_skips_gracefully(self, tmp_dir):
        """Entry for a project with no map → skipped, no crash."""
        llm = _FakeLLMClient(responses=["should not be called"])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        entry = _mk_entry(project="nonexistent-proj")
        await sys._integrate_entries_into_maps([entry])

        # LLM was NOT called — no map to integrate into.
        assert len(llm.calls) == 0
        sys._store.close()

    @pytest.mark.asyncio
    async def test_empty_project_entries_skipped(self, tmp_dir):
        """Entries with empty project → filtered out, no LLM call."""
        llm = _FakeLLMClient(responses=["should not be called"])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        entry = _mk_entry(project="")
        await sys._integrate_entries_into_maps([entry])

        assert len(llm.calls) == 0
        sys._store.close()

    @pytest.mark.asyncio
    async def test_llm_timeout_handled_gracefully(self, tmp_dir):
        """LLM timeout → logged warning, no crash, entries unaffected."""
        async def slow_complete(prompt, kwargs):
            await asyncio.sleep(5)
            return "too slow"

        llm = _FakeLLMClient(side_effect=slow_complete)
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        initial_map = "# Project: test-proj\n\n## Current State\nActive.\n"
        await sys.create_map("test-proj", initial_map, project_dir=tmp_dir)

        entry = _mk_entry(project="test-proj")
        original = sys._integrate_project_entries

        async def fast_timeout(project, entries):
            import unittest.mock
            with unittest.mock.patch(
                "mesh.memory.system_v2.asyncio.wait_for",
                side_effect=asyncio.TimeoutError,
            ):
                await original(project, entries)

        sys._integrate_project_entries = fast_timeout
        # Should not raise.
        await sys._integrate_entries_into_maps([entry])

        # Map unchanged.
        content = await sys.get_map("test-proj")
        assert content == initial_map
        sys._store.close()

    @pytest.mark.asyncio
    async def test_no_updates_needed_leaves_map_unchanged(self, tmp_dir):
        """LLM returns 'No updates needed.' → map not modified."""
        llm = _FakeLLMClient(responses=["No updates needed."])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        initial_map = "# Project: test-proj\n\n## Current State\nActive.\n"
        await sys.create_map("test-proj", initial_map, project_dir=tmp_dir)

        entry = _mk_entry(project="test-proj")
        await sys._integrate_entries_into_maps([entry])

        content = await sys.get_map("test-proj")
        assert content == initial_map
        sys._store.close()


class TestApplyMapEditReplaceAll:
    """Tests for the replace_all parameter on apply_map_edit."""

    @pytest.mark.asyncio
    async def test_replace_all_false_rejects_multiple_matches(self, tmp_dir):
        """Default (replace_all=False) errors on multiple matches."""
        llm = _FakeLLMClient()
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        content = "# Project: test\n\nfoo bar\nfoo baz\n"
        await sys.create_map("test", content, project_dir=tmp_dir)

        result = await sys.apply_map_edit("test", "foo", "qux")
        assert "Error" in result
        assert "2 locations" in result
        sys._store.close()

    @pytest.mark.asyncio
    async def test_replace_all_true_replaces_all_matches(self, tmp_dir):
        """replace_all=True replaces every occurrence."""
        llm = _FakeLLMClient()
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        content = "# Project: test\n\nfoo bar\nfoo baz\n"
        await sys.create_map("test", content, project_dir=tmp_dir)

        result = await sys.apply_map_edit("test", "foo", "qux", replace_all=True)
        assert "successfully" in result
        assert "2 replacement" in result

        updated = await sys.get_map("test")
        assert "foo" not in updated
        assert updated.count("qux") == 2
        sys._store.close()


# ── Full Pipeline: end-to-end formation → integration ─────────────────


class _DispatchingLLMClient:
    """Fake LLM that dispatches to different responses based on prompt content.

    Formation calls contain the canonical unified-fold wording.
    Integration calls contain 'maintaining a project knowledge map' (from system_v2.py).
    Bootstrap calls contain 'bootstrap' or 'You are initializing a project' (from NEW_PROJECT_MAP_PROMPT).
    """

    def __init__(
        self,
        segmenter_response: str,
        integration_responses: dict[str, str] | None = None,
        bootstrap_response: str | None = None,
        integration_side_effect: Exception | None = None,
    ):
        self._segmenter_response = segmenter_response
        self._integration_responses = integration_responses or {}
        self._bootstrap_response = bootstrap_response
        self._integration_side_effect = integration_side_effect
        self.segmenter_calls: list[str] = []
        self.integration_calls: list[str] = []
        self.bootstrap_calls: list[str] = []
        self.all_calls: list[str] = []

    async def complete(self, prompt: str, **kwargs) -> str:
        self.all_calls.append(prompt)
        if "memory-formation step of a unified fold pass" in prompt:
            self.segmenter_calls.append(prompt)
            return self._segmenter_response
        if "maintaining a project knowledge map" in prompt:
            self.integration_calls.append(prompt)
            if self._integration_side_effect:
                raise self._integration_side_effect
            for proj_key, response in self._integration_responses.items():
                if proj_key in prompt:
                    return response
            return "No updates needed."
        if "A new project has been identified" in prompt:
            self.bootstrap_calls.append(prompt)
            return self._bootstrap_response or (
                "# Project: new-proj\n\n## Architecture\nTBD\n\n"
                "## Current State\nBootstrapped.\n\n## Open Issues\nNone yet.\n"
            )
        return ""


class TestFullPipeline:
    """End-to-end tests: conversation → form_un_formed → segmenter → persist → integration → map."""

    @pytest.mark.asyncio
    async def test_full_pipeline_end_to_end(self, tmp_dir):
        """Feed multi-turn history → segmenter produces entries → entries integrated into map.

        Verifies: entries persisted with correct project, cursor advanced,
        map content updated with entry information.
        """
        seg_response = _mk_segments_json([
            _mk_seg(0, 4, worthwhile=True, project="mesh-infra",
                    summary="Fixed socket path security issue",
                    retrieval_key="socket path hardening",
                    topic_label="socket-security", score=8, tags=["security"]),
            _mk_seg(5, 9, worthwhile=True, project="mesh-infra",
                    summary="Added env var allowlist for subprocesses",
                    retrieval_key="subprocess env allowlist",
                    topic_label="env-allowlist", score=7, tags=["security"]),
        ])
        integration_response = (
            '<append_to name="## Key Decisions">\n'
            "- **Socket path** — moved to ~/.mesh/sockets/ with mode 0700\n"
            "- **Env allowlist** — subprocess env vars filtered through frozenset\n"
            "</append_to>"
        )
        llm = _DispatchingLLMClient(
            segmenter_response=seg_response,
            integration_responses={"mesh-infra": integration_response},
        )
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        initial_map = (
            "# Project: mesh-infra\n\n"
            "## Architecture\nRouter + agents + memory.\n\n"
            "## Key Decisions\n- **V3 formation** — DeepSeek segmenter\n\n"
            "## Open Issues\nNone.\n"
        )
        await sys.create_map("mesh-infra", initial_map, project_dir=tmp_dir)

        turns = [_turn(i, f"discussing security fix {i}") for i in range(10)]
        n = await sys.form_un_formed(turns, "time-based")

        assert n == 2
        await asyncio.sleep(0.15)

        assert len(sys._pool) == 2
        assert sys._pool[0].project == "mesh-infra"
        assert sys._pool[1].project == "mesh-infra"
        assert "socket path" in sys._pool[0].summary.lower()
        assert "env var" in sys._pool[1].summary.lower() or "allowlist" in sys._pool[1].summary.lower()

        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 10

        assert len(llm.segmenter_calls) == 1
        assert len(llm.integration_calls) == 1

        updated_map = await sys.get_map("mesh-infra")
        assert "Socket path" in updated_map
        assert "Env allowlist" in updated_map
        assert "V3 formation" in updated_map  # original content preserved
        sys._store.close()

    @pytest.mark.asyncio
    async def test_cold_start_bootstrap_then_integration(self, tmp_dir):
        """No pre-existing map. Segmenter assigns to unknown project.

        Verify: bootstrap creates skeleton, then integration fills it in.
        The fire-and-forget task must see the skeleton that bootstrap created.
        """
        seg_response = _mk_segments_json([
            _mk_seg(0, 4, worthwhile=True, project="new-proj",
                    summary="Started work on new project X",
                    retrieval_key="new project X bootstrap",
                    topic_label="project-bootstrap", score=6, tags=["setup"]),
        ])
        bootstrap_map = (
            "# Project: new-proj\n\n"
            "## Architecture\nTBD\n\n"
            "## Current State\nJust started.\n\n"
            "## Open Issues\nNone yet.\n"
        )
        integration_response = (
            '<section name="## Current State">\n'
            "Started work on new project X. Initial setup in progress.\n"
            "</section>"
        )
        llm = _DispatchingLLMClient(
            segmenter_response=seg_response,
            integration_responses={"new-proj": integration_response},
            bootstrap_response=bootstrap_map,
        )
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)
        sys._active_project_dir = tmp_dir

        turns = [_turn(i, f"setting up project {i}") for i in range(5)]
        n = await sys.form_un_formed(turns, "time-based")

        assert n == 1
        await asyncio.sleep(0.15)

        assert len(sys._pool) == 1
        assert sys._pool[0].project == "new-proj"

        assert len(llm.bootstrap_calls) == 1

        updated_map = await sys.get_map("new-proj")
        assert updated_map is not None, "Map should exist after bootstrap + integration"
        assert "Initial setup in progress" in updated_map
        assert len(llm.integration_calls) == 1
        sys._store.close()

    @pytest.mark.asyncio
    async def test_multi_project_batch(self, tmp_dir):
        """Conversation spanning 2 projects in one batch.

        Pre-create maps for both. Verify: each map independently updated,
        separate LLM calls per project, content matches relevant entries only.
        """
        import os

        seg_response = _mk_segments_json([
            _mk_seg(0, 4, worthwhile=True, project="proj-alpha",
                    summary="Refactored the router",
                    retrieval_key="router refactor",
                    topic_label="router-work", score=7, tags=["refactor"]),
            _mk_seg(5, 9, worthwhile=True, project="proj-beta",
                    summary="Fixed memory cursor bug",
                    retrieval_key="cursor bug fix",
                    topic_label="cursor-fix", score=8, tags=["bugfix"]),
        ])
        llm = _DispatchingLLMClient(
            segmenter_response=seg_response,
            integration_responses={
                "proj-alpha": (
                    '<append_to name="## Current State">\n'
                    "- Router refactored for clarity\n"
                    "</append_to>"
                ),
                "proj-beta": (
                    '<append_to name="## Current State">\n'
                    "- Memory cursor bug fixed\n"
                    "</append_to>"
                ),
            },
        )
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        dir_a = os.path.join(tmp_dir, "proj-alpha")
        dir_b = os.path.join(tmp_dir, "proj-beta")
        os.makedirs(dir_a, exist_ok=True)
        os.makedirs(dir_b, exist_ok=True)
        await sys.create_map(
            "proj-alpha",
            "# Project: proj-alpha\n\n## Current State\nActive.\n",
            project_dir=dir_a,
        )
        await sys.create_map(
            "proj-beta",
            "# Project: proj-beta\n\n## Current State\nActive.\n",
            project_dir=dir_b,
        )

        turns = [_turn(i, f"working on multi-project task {i}") for i in range(10)]
        n = await sys.form_un_formed(turns, "time-based")

        assert n == 2
        await asyncio.sleep(0.15)

        projects = {e.project for e in sys._pool}
        assert projects == {"proj-alpha", "proj-beta"}

        assert len(llm.integration_calls) == 2

        map_a = await sys.get_map("proj-alpha")
        assert "Router refactored" in map_a
        assert "cursor bug" not in map_a.lower()

        map_b = await sys.get_map("proj-beta")
        assert "cursor bug fixed" in map_b
        assert "router" not in map_b.lower() or "Router refactored" not in map_b

        sys._store.close()

    @pytest.mark.asyncio
    async def test_cursor_advances_despite_integration_failure(self, tmp_dir):
        """Segmenter succeeds, entries persisted, but integration raises.

        Verify: cursor still advances, entries in pool, maps NOT updated.
        """
        seg_response = _mk_segments_json([
            _mk_seg(0, 4, worthwhile=True, project="fail-proj",
                    summary="Work that will fail to integrate",
                    retrieval_key="integration failure test",
                    topic_label="fail-test", score=5, tags=["test"]),
        ])
        llm = _DispatchingLLMClient(
            segmenter_response=seg_response,
            integration_side_effect=RuntimeError("LLM backend down"),
        )
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        initial_map = "# Project: fail-proj\n\n## Current State\nPre-test.\n"
        await sys.create_map("fail-proj", initial_map, project_dir=tmp_dir)

        turns = [_turn(i) for i in range(5)]
        n = await sys.form_un_formed(turns, "time-based")

        assert n == 1
        await asyncio.sleep(0.15)

        cursor_idx, _ = sys._store.get_formation_cursor()
        assert cursor_idx == 5, "Cursor must advance even when integration fails"

        assert len(sys._pool) == 1
        assert sys._pool[0].project == "fail-proj"

        map_content = await sys.get_map("fail-proj")
        assert map_content == initial_map, "Map should be unchanged after integration failure"
        sys._store.close()

    @pytest.mark.asyncio
    async def test_idempotent_rerun(self, tmp_dir):
        """Run form_un_formed twice on same window. Second is a no-op."""
        seg_response = _mk_segments_json([
            _mk_seg(0, 4, worthwhile=True, project="idem-proj",
                    summary="Some work",
                    retrieval_key="idempotency test",
                    topic_label="idem-test", score=6, tags=["test"]),
        ])
        llm = _DispatchingLLMClient(
            segmenter_response=seg_response,
            integration_responses={"idem-proj": "No updates needed."},
        )
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        await sys.create_map(
            "idem-proj",
            "# Project: idem-proj\n\n## Current State\nActive.\n",
            project_dir=tmp_dir,
        )

        turns = [_turn(i) for i in range(5)]

        # First run: creates entries
        n1 = await sys.form_un_formed(turns, "time-based")
        assert n1 == 1
        await asyncio.sleep(0.15)

        calls_after_first = len(llm.all_calls)
        pool_after_first = len(sys._pool)
        cursor_after_first, _ = sys._store.get_formation_cursor()

        # Second run: same history, cursor already past these turns
        n2 = await sys.form_un_formed(turns, "time-based")
        assert n2 == 0, "Second run should produce zero entries"

        assert len(llm.all_calls) == calls_after_first, "No additional LLM calls on second run"
        assert len(sys._pool) == pool_after_first, "No new entries on second run"
        cursor_after_second, _ = sys._store.get_formation_cursor()
        assert cursor_after_second == cursor_after_first, "Cursor unchanged on second run"

        sys._store.close()


# ── Relevance-Based Map Injection ────────────────────────────────────


class TestMapSummaryExtraction:
    def test_extracts_summary_section(self):
        content = (
            "# Project Map\n\n"
            "## Summary\n\n"
            "This is a test project for mesh agents.\n"
            "It handles memory formation.\n\n"
            "## Goals\n\n"
            "- Build something great\n"
        )
        result = MemorySystemV2._extract_map_summary(content)
        assert result == "This is a test project for mesh agents. It handles memory formation."

    def test_returns_none_when_no_summary_section(self):
        content = "# Project Map\n\n## Goals\n\n- Do stuff\n"
        result = MemorySystemV2._extract_map_summary(content)
        assert result is None

    def test_stops_at_next_heading(self):
        content = (
            "## Summary\n"
            "Line one.\n"
            "Line two.\n"
            "## Architecture\n"
            "Should not appear.\n"
        )
        result = MemorySystemV2._extract_map_summary(content)
        assert result == "Line one. Line two."

    def test_skips_blank_lines(self):
        content = "## Summary\n\nSpaced out.\n\n\n## Next\n"
        result = MemorySystemV2._extract_map_summary(content)
        assert result == "Spaced out."


class TestMapFilePath:
    @pytest.mark.asyncio
    async def test_returns_central_path(self, tmp_dir):
        import os
        from unittest.mock import patch
        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        with patch("mesh.paths.MAPS_DIR", os.path.join(tmp_dir, "maps")):
            path = sys._map_file_path("test-project")
            assert path.endswith("maps/test-project.md")

        sys._store.close()

    @pytest.mark.asyncio
    async def test_migrates_from_old_location(self, tmp_dir):
        import os
        from unittest.mock import patch

        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        maps_dir = os.path.join(tmp_dir, "maps")
        old_dir = os.path.join(tmp_dir, "old_project")
        os.makedirs(old_dir, exist_ok=True)
        old_path = os.path.join(old_dir, "PROJECT_MAP.md")
        with open(old_path, "w") as f:
            f.write("# Old Map\n## Summary\nOld content.\n")

        sys._active_project = "test-project"
        sys._active_project_dir = old_dir

        with patch("mesh.paths.MAPS_DIR", maps_dir):
            path = sys._map_file_path("test-project")
            assert os.path.exists(path)
            assert path == os.path.join(maps_dir, "test-project.md")
            with open(path) as f:
                assert "Old content." in f.read()

        sys._store.close()


class TestCreateMapCentral:
    @pytest.mark.asyncio
    async def test_creates_in_central_directory(self, tmp_dir):
        import os
        from unittest.mock import patch

        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder()
        sys = _build_memory_system(tmp_dir, llm, embedder)

        maps_dir = os.path.join(tmp_dir, "maps")
        content = "# Test\n## Summary\nA test map.\n## Goals\n- Ship it\n"

        with patch("mesh.paths.MAPS_DIR", maps_dir):
            ok = await sys.create_map("central-proj", content)
            assert ok
            central = os.path.join(maps_dir, "central-proj.md")
            assert os.path.exists(central)
            with open(central) as f:
                assert f.read() == content

        await asyncio.sleep(0.1)
        sys._store.close()


class TestSelectRelevantMaps:
    @pytest.mark.asyncio
    async def test_returns_top_k(self, tmp_dir):
        import os
        from unittest.mock import patch

        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder(dim=8)
        sys = _build_memory_system(tmp_dir, llm, embedder)

        maps_dir = os.path.join(tmp_dir, "maps")
        os.makedirs(maps_dir, exist_ok=True)

        # Create 3 maps with summaries and embeddings
        for name in ["alpha", "beta", "gamma"]:
            content = f"# {name}\n## Summary\n{name} project does things.\n"
            with open(os.path.join(maps_dir, f"{name}.md"), "w") as f:
                f.write(content)
            sys._store.create_map(f"map-{name}", name, content)
            emb = await embedder.embed_to_array(f"{name} project does things.")
            sys._store.update_map_summary(name, f"{name} project does things.", emb)

        with patch("mesh.paths.MAPS_DIR", maps_dir):
            results = await sys.select_relevant_maps("alpha project does things.", k=2, min_score=-1.0)

        assert len(results) <= 2
        names_returned = [r[0] for r in results]
        assert len(names_returned) > 0
        # Each result is (name, score, content)
        for name, score, content in results:
            assert isinstance(score, float)
            assert content.startswith("#")

        sys._store.close()

    @pytest.mark.asyncio
    async def test_respects_min_score(self, tmp_dir):
        import os
        from unittest.mock import patch

        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder(dim=8)
        sys = _build_memory_system(tmp_dir, llm, embedder)

        maps_dir = os.path.join(tmp_dir, "maps")
        os.makedirs(maps_dir, exist_ok=True)

        content = "# test\n## Summary\nVery specific topic.\n"
        with open(os.path.join(maps_dir, "test.md"), "w") as f:
            f.write(content)
        sys._store.create_map("map-test", "test", content)
        emb = await embedder.embed_to_array("Very specific topic.")
        sys._store.update_map_summary("test", "Very specific topic.", emb)

        with patch("mesh.paths.MAPS_DIR", maps_dir):
            # With an impossibly high min_score, nothing should match
            results = await sys.select_relevant_maps("unrelated query", k=2, min_score=0.99)

        assert len(results) == 0

        sys._store.close()

    @pytest.mark.asyncio
    async def test_empty_when_no_embeddings(self, tmp_dir):
        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder(dim=8)
        sys = _build_memory_system(tmp_dir, llm, embedder)

        results = await sys.select_relevant_maps("some query", k=2)
        assert results == []

        sys._store.close()

    @pytest.mark.asyncio
    async def test_empty_when_no_context(self, tmp_dir):
        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder(dim=8)
        sys = _build_memory_system(tmp_dir, llm, embedder)

        results = await sys.select_relevant_maps("   ", k=2)
        assert results == []

        sys._store.close()


class TestSummaryUpdatedOnIntegration:
    @pytest.mark.asyncio
    async def test_summary_and_embedding_updated_directly(self, tmp_dir):
        import os
        from unittest.mock import patch

        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder(dim=8)
        sys = _build_memory_system(tmp_dir, llm, embedder)

        maps_dir = os.path.join(tmp_dir, "maps")
        os.makedirs(maps_dir, exist_ok=True)

        content = "# Test\n## Summary\nA project about widgets.\n## Goals\n- Build widgets\n"
        with open(os.path.join(maps_dir, "test-proj.md"), "w") as f:
            f.write(content)
        sys._store.create_map("map-test", "test-proj", content)

        with patch("mesh.paths.MAPS_DIR", maps_dir):
            await sys._update_map_summary_and_embedding("test-proj", content)

        rows = sys._store.list_map_embeddings()
        found = [r for r in rows if r[0] == "test-proj"]
        assert len(found) == 1
        _name, summary, emb_blob = found[0]
        assert summary == "A project about widgets."
        assert emb_blob is not None

        sys._store.close()

    @pytest.mark.asyncio
    async def test_summary_fallback_when_no_section(self, tmp_dir):
        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder(dim=8)
        sys = _build_memory_system(tmp_dir, llm, embedder)

        content = "# Project\nThis is the first paragraph of text.\n\n## Architecture\nArch stuff.\n"
        sys._store.create_map("map-nosummary", "nosummary", content)

        await sys._update_map_summary_and_embedding("nosummary", content)

        rows = sys._store.list_map_embeddings()
        found = [r for r in rows if r[0] == "nosummary"]
        assert len(found) == 1
        _name, summary, emb_blob = found[0]
        assert "first paragraph" in summary
        assert emb_blob is not None

        sys._store.close()


class TestRenderRelevantMapsBlock:
    @pytest.mark.asyncio
    async def test_falls_back_to_static_when_no_embeddings(self, tmp_dir):
        import os
        from unittest.mock import patch

        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder(dim=8)
        sys = _build_memory_system(tmp_dir, llm, embedder)

        maps_dir = os.path.join(tmp_dir, "maps")
        os.makedirs(maps_dir, exist_ok=True)

        content = "# Fallback\n## Summary\nFallback map.\n"
        with open(os.path.join(maps_dir, "fallback.md"), "w") as f:
            f.write(content)
        sys._active_project = "fallback"

        with patch("mesh.paths.MAPS_DIR", maps_dir):
            block = await sys.render_relevant_maps_block("some context")
            # Falls back to render_maps_block which uses _active_project
            assert "fallback" in block.lower()

        sys._store.close()

    @pytest.mark.asyncio
    async def test_renders_relevant_maps_with_scores(self, tmp_dir):
        import os
        from unittest.mock import patch

        llm = _FakeLLMClient(responses=[])
        embedder = _FakeEmbedder(dim=8)
        sys = _build_memory_system(tmp_dir, llm, embedder)

        maps_dir = os.path.join(tmp_dir, "maps")
        os.makedirs(maps_dir, exist_ok=True)

        content = "# MyProj\n## Summary\nA relevant project.\n"
        with open(os.path.join(maps_dir, "myproj.md"), "w") as f:
            f.write(content)
        sys._store.create_map("map-myproj", "myproj", content)
        emb = await embedder.embed_to_array("A relevant project.")
        sys._store.update_map_summary("myproj", "A relevant project.", emb)

        with patch("mesh.paths.MAPS_DIR", maps_dir):
            block = await sys.render_relevant_maps_block("A relevant project.", min_score=-1.0)

        assert 'project="myproj"' in block
        assert 'relevance="' in block

        sys._store.close()


# ── Lazy Greedy FL + Two-Stage TOC ───────────────────────────────────


class TestLazyGreedyFL:
    """Tests for the lazy greedy Facility Location algorithm."""

    def test_selects_k_items(self):
        from mesh.memory.selection import lazy_greedy_fl, _build_sim_matrix
        rng = np.random.RandomState(42)
        embs = [rng.randn(8).astype(np.float32) for _ in range(20)]
        sim = _build_sim_matrix(embs)
        selected = lazy_greedy_fl(sim, k=5)
        assert len(selected) == 5
        assert len(set(selected)) == 5  # no duplicates

    def test_selects_all_when_k_exceeds_n(self):
        from mesh.memory.selection import lazy_greedy_fl, _build_sim_matrix
        rng = np.random.RandomState(42)
        embs = [rng.randn(8).astype(np.float32) for _ in range(3)]
        sim = _build_sim_matrix(embs)
        selected = lazy_greedy_fl(sim, k=10)
        assert len(selected) == 3

    def test_empty_input(self):
        from mesh.memory.selection import lazy_greedy_fl
        selected = lazy_greedy_fl(np.array([]).reshape(0, 0), k=5)
        assert selected == []

    def test_picks_diverse_clusters(self):
        """Given 3 tight clusters, FL should pick at least one from each."""
        from mesh.memory.selection import lazy_greedy_fl, _build_sim_matrix
        rng = np.random.RandomState(99)
        cluster_centers = [
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
            np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
        ]
        embs = []
        labels = []
        for ci, center in enumerate(cluster_centers):
            for _ in range(10):
                noise = rng.randn(8).astype(np.float32) * 0.05
                embs.append(center + noise)
                labels.append(ci)
        sim = _build_sim_matrix(embs)
        selected = lazy_greedy_fl(sim, k=6)
        selected_labels = {labels[i] for i in selected}
        assert selected_labels == {0, 1, 2}, (
            f"Expected all 3 clusters, got labels {selected_labels}"
        )

    def test_single_item(self):
        from mesh.memory.selection import lazy_greedy_fl, _build_sim_matrix
        embs = [np.array([1, 0, 0], dtype=np.float32)]
        sim = _build_sim_matrix(embs)
        selected = lazy_greedy_fl(sim, k=1)
        assert selected == [0]


class TestBuildTocTwoStage:
    """Tests for the two-stage TOC pipeline (relevance + FL diversity)."""

    @pytest.mark.asyncio
    async def test_fl_selection_produces_diverse_results(self, tmp_dir):
        """Given clustered entries, TOC should pick representatives from each cluster."""
        embedder = _FakeEmbedder(dim=8)
        llm = _FakeLLMClient(responses=[])
        sys = _build_memory_system(tmp_dir, llm, embedder)
        now = datetime.now(timezone.utc)

        rng = np.random.RandomState(42)
        centers = [
            np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
            np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32),
        ]
        entries = []
        for ci, center in enumerate(centers):
            for j in range(15):
                noise = rng.randn(8).astype(np.float32) * 0.05
                emb = center + noise
                entry = MemoryEntry(
                    id=f"e_{ci}_{j}",
                    created_at=now,
                    summary=f"Cluster {ci} entry {j}",
                    reflection="",
                    trace="",
                    trigger="",
                    retrieval_key=f"cluster-{ci}-item-{j}",
                    retrieval_key_embedding=emb,
                    project="test",
                )
                entries.append(entry)

        sys._pool = entries
        sys._active_project = None

        query_emb = np.array([0.5, 0.3, 0.2, 0.4, 0.1, 0.1, 0.3, 0.1], dtype=np.float32)

        async def fake_embed(text):
            return query_emb
        sys._embedder.embed_to_array = fake_embed

        toc = await sys.build_toc(
            query_text="test query",
            k=6,
        )

        assert len(toc) == 6
        cluster_ids = {e.retrieval_key.split("-")[1] for e in toc}
        assert len(cluster_ids) >= 2, (
            f"Expected entries from multiple clusters, got {cluster_ids}"
        )
        sys._store.close()

    @pytest.mark.asyncio
    async def test_fallback_to_recency_without_embeddings(self, tmp_dir):
        """No query text → falls back to recency ordering."""
        embedder = _FakeEmbedder(dim=8)
        llm = _FakeLLMClient(responses=[])
        sys = _build_memory_system(tmp_dir, llm, embedder)

        now = datetime.now(timezone.utc)
        entries = []
        for i in range(5):
            entry = MemoryEntry(
                id=f"e_{i}",
                created_at=now - timedelta(hours=5 - i),
                summary=f"Entry {i}",
                reflection="",
                trace="",
                trigger="",
                retrieval_key=f"key-{i}",
            )
            entries.append(entry)
        sys._pool = entries
        sys._active_project = None

        toc = await sys.build_toc(query_text=None, k=3)

        assert len(toc) == 3
        assert toc[0].id == "e_4"
        assert toc[1].id == "e_3"
        assert toc[2].id == "e_2"
        sys._store.close()

    @pytest.mark.asyncio
    async def test_small_pool_skips_fl(self, tmp_dir):
        """When pool <= k, FL is skipped, all entries returned."""
        embedder = _FakeEmbedder(dim=8)
        llm = _FakeLLMClient(responses=[])
        sys = _build_memory_system(tmp_dir, llm, embedder)
        now = datetime.now(timezone.utc)

        rng = np.random.RandomState(42)
        entries = []
        for i in range(3):
            emb = rng.randn(8).astype(np.float32)
            entry = MemoryEntry(
                id=f"e_{i}",
                created_at=now,
                summary=f"Entry {i}",
                reflection="",
                trace="",
                trigger="",
                retrieval_key=f"key-{i}",
                retrieval_key_embedding=emb,
            )
            entries.append(entry)
        sys._pool = entries
        sys._active_project = None

        async def fake_embed(text):
            return rng.randn(8).astype(np.float32)
        sys._embedder.embed_to_array = fake_embed

        toc = await sys.build_toc(query_text="test", k=5)

        assert len(toc) == 3
        sys._store.close()

    @pytest.mark.asyncio
    async def test_context_text_used_over_query_text(self, tmp_dir):
        """context_text is preferred for embedding when both provided."""
        embedder = _FakeEmbedder(dim=8)
        llm = _FakeLLMClient(responses=[])
        sys = _build_memory_system(tmp_dir, llm, embedder)
        now = datetime.now(timezone.utc)

        rng = np.random.RandomState(42)
        entries = []
        for i in range(5):
            emb = rng.randn(8).astype(np.float32)
            entry = MemoryEntry(
                id=f"e_{i}",
                created_at=now,
                summary=f"Entry {i}",
                reflection="",
                trace="",
                trigger="",
                retrieval_key=f"key-{i}",
                retrieval_key_embedding=emb,
            )
            entries.append(entry)
        sys._pool = entries
        sys._active_project = None

        embed_calls = []
        original_embed = embedder.embed_to_array

        async def tracking_embed(text):
            embed_calls.append(text)
            return await original_embed(text)
        sys._embedder.embed_to_array = tracking_embed

        toc = await sys.build_toc(
            query_text="short query",
            k=3,
            context_text="broader conversation context with more detail",
        )

        assert len(toc) > 0
        assert any("broader conversation" in c for c in embed_calls)
        assert not any(c == "short query" for c in embed_calls)
        sys._store.close()

    @pytest.mark.asyncio
    async def test_restriction_size_limits_candidates(self, tmp_dir):
        """restriction_size controls how many candidates enter Stage 2."""
        embedder = _FakeEmbedder(dim=8)
        llm = _FakeLLMClient(responses=[])
        sys = _build_memory_system(tmp_dir, llm, embedder)
        now = datetime.now(timezone.utc)

        rng = np.random.RandomState(42)
        entries = []
        for i in range(50):
            emb = rng.randn(8).astype(np.float32)
            entry = MemoryEntry(
                id=f"e_{i}",
                created_at=now,
                summary=f"Entry {i}",
                reflection="",
                trace="",
                trigger="",
                retrieval_key=f"key-{i}",
                retrieval_key_embedding=emb,
            )
            entries.append(entry)
        sys._pool = entries
        sys._active_project = None

        async def fake_embed(text):
            return rng.randn(8).astype(np.float32)
        sys._embedder.embed_to_array = fake_embed

        toc = await sys.build_toc(
            query_text="test",
            k=5,
            restriction_size=10,
        )

        assert len(toc) == 5
        sys._store.close()
