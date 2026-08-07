"""Tests for the dry-run memory formation harness.

These tests deliberately avoid real LLM calls — every test either uses
the offline mode (which skips DeepSeek entirely) or monkey-patches
``_llm_chat`` to return canned responses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.memory_dryrun import replay, strategies
from benchmark.memory_dryrun.replay import Turn
from benchmark.memory_dryrun.run import run_pipeline, segment_metrics


# ──────────────────────────────────────────────────────────────────────
# replay.py
# ──────────────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_replay_jsonl_basic(tmp_path: Path) -> None:
    fixture = tmp_path / "agent-bob.json"
    _write_jsonl(
        fixture,
        [
            {
                "message": {
                    "from_node": "user:testuser",
                    "to_node": "agent:sysadmin:bob",
                    "type": "message",
                    "content": "hey bob",
                    "id": "msg-1",
                    "timestamp": "2026-04-26T22:00:00Z",
                    "metadata": {},
                },
                "direction": "incoming",
            },
            {
                "message": {
                    "from_node": "agent:sysadmin:bob",
                    "to_node": "user:testuser",
                    "type": "message",
                    "content": "hello",
                    "id": "msg-2",
                    "timestamp": "2026-04-26T22:00:01Z",
                    "metadata": {},
                },
                "direction": "outgoing",
            },
            # malformed line should be skipped
            {"not_a_message": True},
            # empty content should be skipped
            {
                "message": {
                    "from_node": "user:testuser",
                    "to_node": "agent:sysadmin:bob",
                    "type": "message",
                    "content": "",
                    "id": "msg-3",
                    "timestamp": "2026-04-26T22:00:02Z",
                    "metadata": {},
                },
                "direction": "incoming",
            },
        ],
    )
    turns = replay.replay_jsonl(fixture)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].role == "agent"
    assert turns[0].content == "hey bob"
    assert turns[1].content == "hello"


def test_replay_jsonl_skips_bad_json(tmp_path: Path) -> None:
    fixture = tmp_path / "agent-x.json"
    fixture.write_text(
        '{"message": {"type": "message", "content": "ok", "from_node": "user:a", "to_node": "agent:b"}, "direction": "incoming"}\n'
        'not json at all\n'
        '{"message": {"type": "message", "content": "also ok", "from_node": "user:a", "to_node": "agent:b"}, "direction": "incoming"}\n'
    )
    turns = replay.replay_jsonl(fixture)
    assert len(turns) == 2


def test_replay_log_basic(tmp_path: Path) -> None:
    log_path = tmp_path / "agent-bob.log"
    log_path.write_text(
        "2026-04-26 22:00:00 [DEBUG] mesh.transport:75 - "
        "RECV [127.0.0.1:7700] type=message from=user:testuser to=agent:sysadmin:bob "
        "id=msg-aaaa... content_preview='hey bob'\n"
        "2026-04-26 22:00:01 [DEBUG] mesh.transport:52 - "
        "SEND [127.0.0.1:7700] type=message from=agent:sysadmin:bob to=user:testuser "
        "id=msg-bbbb... content_preview='hello'\n"
        "2026-04-26 22:00:02 [DEBUG] mesh.llm:797 - LLM response: len=42\n"
    )
    turns = replay.replay_log(log_path)
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].content == "hey bob"
    assert turns[0].metadata.get("log_truncated") is True
    assert turns[1].role == "agent"


# ──────────────────────────────────────────────────────────────────────
# strategies.py — segmenters
# ──────────────────────────────────────────────────────────────────────


def _t(role: str, content: str = "x", idx: int = 0) -> Turn:
    if role == "user":
        f = "user:testuser"
        to = "agent:sysadmin:bob"
    else:
        f = "agent:sysadmin:bob"
        to = "user:testuser"
    return Turn(
        timestamp=f"2026-04-26T22:00:{idx:02d}Z",
        from_node=f,
        to_node=to,
        content=content,
        role=role,
        direction="incoming" if role == "user" else "outgoing",
        msg_id=f"msg-{idx:04d}",
    )


def test_current_segmenter_offline_returns_one_segment() -> None:
    """In offline mode the classifier always says SAME, so we get one segment."""
    seg = strategies.CurrentSegmenter(offline=True)
    turns = [_t("user", f"q{i}", i) for i in range(10)]
    out = seg.segment(turns)
    assert len(out) == 1
    assert out[0].turn_count == 10


def test_current_segmenter_force_flush_at_max() -> None:
    """Hard backstop should split into multiple segments."""
    seg = strategies.CurrentSegmenter(offline=True, max_segment_turns=5, classify_every_n=1)
    turns = [_t("user", f"q{i}", i) for i in range(12)]
    out = seg.segment(turns)
    assert len(out) >= 2
    assert all(s.turn_count <= 5 for s in out)
    assert sum(s.turn_count for s in out) == 12


def test_batch_segmenter_fixed_size() -> None:
    seg = strategies.BatchSegmenter(size=4)
    turns = [_t("user", f"q{i}", i) for i in range(10)]
    out = seg.segment(turns)
    assert len(out) == 3
    assert [s.turn_count for s in out] == [4, 4, 2]
    assert out[0].start_idx == 0 and out[0].end_idx == 3
    assert out[2].start_idx == 8 and out[2].end_idx == 9


def test_batch_segmenter_empty_input() -> None:
    assert strategies.BatchSegmenter().segment([]) == []


def test_batch_segmenter_time_window() -> None:
    """Time-window mode flushes when timestamps span > N seconds."""
    seg = strategies.BatchSegmenter(time_window_seconds=60)
    turns = [
        Turn(timestamp="2026-04-26T22:00:00Z", from_node="user:a", to_node="agent:b",
             content="a", role="user", direction="incoming"),
        Turn(timestamp="2026-04-26T22:00:30Z", from_node="user:a", to_node="agent:b",
             content="b", role="user", direction="incoming"),
        Turn(timestamp="2026-04-26T22:02:00Z", from_node="user:a", to_node="agent:b",
             content="c", role="user", direction="incoming"),
    ]
    out = seg.segment(turns)
    assert len(out) == 2
    assert out[0].turn_count == 2
    assert out[1].turn_count == 1


def _llm_seg_payload(segments: list[dict]) -> str:
    """Helper: build a JSON payload as the LLM would emit it."""
    defaults = {
        "topic_label": "topic",
        "worthwhile": True,
        "score": 7,
        "retrieval_key": "what happened",
        "summary": "the thing happened",
        "tags": ["a", "b"],
        "outcome": "success",
    }
    full = []
    for s in segments:
        merged = {**defaults, **s}
        full.append(merged)
    return json.dumps({"segments": full})


def test_llm_segmenter_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-window structured-output run produces segments with metadata."""
    payload = _llm_seg_payload([
        {"start_turn": 0, "end_turn": 2, "topic_label": "first", "score": 8},
        {"start_turn": 3, "end_turn": 4, "topic_label": "second", "worthwhile": False},
    ])
    monkeypatch.setattr(strategies, "_llm_chat", lambda *a, **k: payload)

    seg = strategies.LLMSegmenter(window_size=80, overlap=0, defer_tail_turns=0)
    turns = [_t("user", f"q{i}", i) for i in range(5)]
    out = seg.segment(turns)

    assert len(out) == 2
    assert out[0].topic_label == "first"
    assert out[0].metadata["worthwhile"] is True
    assert out[0].metadata["score"] == 8
    assert out[0].metadata["retrieval_key"] == "what happened"
    assert out[1].topic_label == "second"
    assert out[1].metadata["worthwhile"] is False
    # global indices match window indices when window_start=0
    assert out[0].start_idx == 0 and out[0].end_idx == 2
    assert out[1].start_idx == 3 and out[1].end_idx == 4


def test_llm_segmenter_parse_failure_with_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """First call returns garbage, second call returns valid JSON; second wins."""
    calls = {"n": 0}

    def fake(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "this is not JSON at all"
        return _llm_seg_payload([{"start_turn": 0, "end_turn": 2}])

    monkeypatch.setattr(strategies, "_llm_chat", fake)

    seg = strategies.LLMSegmenter(window_size=80, overlap=0, defer_tail_turns=0)
    turns = [_t("user", f"q{i}", i) for i in range(3)]
    out = seg.segment(turns)
    assert len(out) == 1
    assert seg.json_parse_failures == 1


def test_llm_segmenter_persistent_parse_failure_drops_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both attempts fail to parse, the window is skipped."""
    monkeypatch.setattr(strategies, "_llm_chat", lambda *a, **k: "garbage")
    seg = strategies.LLMSegmenter(window_size=80, overlap=0, defer_tail_turns=0)
    turns = [_t("user", f"q{i}", i) for i in range(3)]
    out = seg.segment(turns)
    assert out == []
    assert seg.json_parse_failures == 2


def test_llm_segmenter_drops_malformed_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema validation drops bad segments without crashing the window."""
    bad_payload = json.dumps({
        "segments": [
            # valid
            {"start_turn": 0, "end_turn": 1, "topic_label": "ok",
             "worthwhile": True, "score": 5, "retrieval_key": "k", "summary": "s",
             "tags": ["x"], "outcome": "success"},
            # missing start_turn
            {"end_turn": 2},
            # end_turn < start_turn
            {"start_turn": 5, "end_turn": 2},
            # end_turn out of range
            {"start_turn": 2, "end_turn": 999},
            # not a dict
            "not-a-segment-dict",
            # valid
            {"start_turn": 2, "end_turn": 3, "topic_label": "also-ok",
             "worthwhile": False, "score": 2, "retrieval_key": "k2", "summary": "s2",
             "tags": [], "outcome": None},
        ]
    })
    monkeypatch.setattr(strategies, "_llm_chat", lambda *a, **k: bad_payload)

    seg = strategies.LLMSegmenter(window_size=80, overlap=0, defer_tail_turns=0)
    turns = [_t("user", f"q{i}", i) for i in range(4)]
    out = seg.segment(turns)
    assert len(out) == 2
    assert seg.malformed_segments_dropped == 4


def test_llm_segmenter_top_level_shape_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level 'segments' missing or wrong type counts as a parse failure."""
    monkeypatch.setattr(strategies, "_llm_chat", lambda *a, **k: '{"foo": 1}')
    seg = strategies.LLMSegmenter(window_size=80, overlap=0, defer_tail_turns=0)
    turns = [_t("user", f"q{i}", i) for i in range(3)]
    out = seg.segment(turns)
    assert out == []
    assert seg.json_parse_failures == 2


def test_llm_segmenter_defer_in_non_final_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segments ending in the trailing defer zone are NOT emitted in non-final windows.

    Window covers 80 turns. Segment ends at relative index 75, defer zone is 65..79.
    Should be deferred; next window with overlap will reclassify.
    """
    # Build 100 turns so we have a non-final window 0 (covers 0..79) and final window 1.
    turns = [_t("user", f"q{i}", i) for i in range(100)]

    # Window 0: emit 0..50 ("early"), defer 51..79 (ends in tail >= 65)
    # Window 1 (start at 60, covers 60..99): the deferred range is now at relative 0..19
    #   so emit it as one segment, then maybe more at the tail.
    payload_w0 = _llm_seg_payload([
        {"start_turn": 0, "end_turn": 50, "topic_label": "early"},
        {"start_turn": 51, "end_turn": 75, "topic_label": "tail-deferred"},
    ])
    # Window 1 will see global turns 60..99 (40 turns) → relative 0..39
    # Reclassifies the deferred turns + extends with new ones.
    payload_w1 = _llm_seg_payload([
        # global 60-79 → relative 0-19; this is the "tail-deferred" with full context
        {"start_turn": 0, "end_turn": 19, "topic_label": "tail-resolved"},
        # global 80-99 → relative 20-39
        {"start_turn": 20, "end_turn": 39, "topic_label": "final-segment"},
    ])

    responses = iter([payload_w0, payload_w1])
    monkeypatch.setattr(strategies, "_llm_chat", lambda *a, **k: next(responses))

    seg = strategies.LLMSegmenter(window_size=80, overlap=20, defer_tail_turns=15)
    out = seg.segment(turns)
    labels = [s.topic_label for s in out]
    # Order matters: early first (from window 0), then tail-resolved (from window 1, was deferred), then final
    assert labels == ["early", "tail-resolved", "final-segment"]
    # Coverage check: ranges should be contiguous and disjoint
    assert out[0].start_idx == 0 and out[0].end_idx == 50
    assert out[1].start_idx == 60 and out[1].end_idx == 79  # the deferred segment, resolved
    assert out[2].start_idx == 80 and out[2].end_idx == 99
    assert seg.deferred_segments == 1


def test_llm_segmenter_no_double_emit_across_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Window 1 must not re-emit segments already covered by window 0.

    If the LLM in window 1 proposes a segment that overlaps an already-emitted
    range from window 0, it must be skipped (or trimmed if it extends past
    the frontier).
    """
    turns = [_t("user", f"q{i}", i) for i in range(100)]

    # Window 0 (covers 0..79): emit "alpha" 0..40 and "beta" 41..60.
    # Anything ending in 65..79 would defer; pick segments with end < 65.
    payload_w0 = _llm_seg_payload([
        {"start_turn": 0, "end_turn": 40, "topic_label": "alpha"},
        {"start_turn": 41, "end_turn": 60, "topic_label": "beta"},
    ])
    # Window 1 starts at 60, covers 60..99 (rel 0..39).
    # LLM in window 1 might re-propose:
    #   rel 0-5  (global 60-65) — entirely overlaps "beta" (which ended at global 60)?
    #     Actually "beta" ended at global 60, frontier = 61. Segment global 60-65 overlaps.
    #     The first turn (60) is already inside "beta"; trim to global 61-65.
    #   rel 6-39 (global 66-99) — fresh, emit as-is.
    payload_w1 = _llm_seg_payload([
        {"start_turn": 0, "end_turn": 5, "topic_label": "beta-revisit"},
        {"start_turn": 6, "end_turn": 39, "topic_label": "gamma"},
    ])

    responses = iter([payload_w0, payload_w1])
    monkeypatch.setattr(strategies, "_llm_chat", lambda *a, **k: next(responses))

    seg = strategies.LLMSegmenter(window_size=80, overlap=20, defer_tail_turns=15)
    out = seg.segment(turns)

    # No double-emit: total turn count across emitted segments == 100 OR less,
    # and ranges must be strictly increasing with no overlap.
    ranges = [(s.start_idx, s.end_idx) for s in out]
    for (s1, e1), (s2, e2) in zip(ranges, ranges[1:]):
        assert s2 > e1, f"overlapping ranges: ({s1},{e1}) and ({s2},{e2})"

    # The "beta-revisit" segment in window 1 starts at global 60; "beta" already
    # claimed up through 60, so beta-revisit must be either skipped or trimmed
    # to start at >= 61. Either way, alpha and beta are still in the output.
    labels = [s.topic_label for s in out]
    assert "alpha" in labels
    assert "beta" in labels
    assert "gamma" in labels


def test_llm_segmenter_final_window_emits_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Final window emits even segments whose relative end is in the tail.

    Otherwise the last few turns of the conversation would never be captured.
    """
    turns = [_t("user", f"q{i}", i) for i in range(40)]

    # 40 turns < window_size=80 so it's a single (final) window.
    # Segment spans 0..39 and ends at the very end (rel 39, defer zone 65..79
    # but window_len=40 so defer is rel 25..39). is_final=True bypasses defer.
    payload = _llm_seg_payload([
        {"start_turn": 0, "end_turn": 39, "topic_label": "all-of-it"},
    ])
    monkeypatch.setattr(strategies, "_llm_chat", lambda *a, **k: payload)

    seg = strategies.LLMSegmenter(window_size=80, overlap=20, defer_tail_turns=15)
    out = seg.segment(turns)
    assert len(out) == 1
    assert out[0].turn_count == 40
    assert seg.deferred_segments == 0


def test_llm_segmenter_init_validates_args() -> None:
    with pytest.raises(ValueError):
        strategies.LLMSegmenter(window_size=80, overlap=80)
    with pytest.raises(ValueError):
        strategies.LLMSegmenter(defer_tail_turns=-1)
    with pytest.raises(ValueError):
        strategies.LLMSegmenter(window_size=80, defer_tail_turns=80)


# ──────────────────────────────────────────────────────────────────────
# strategies.py — LLMFormer
# ──────────────────────────────────────────────────────────────────────


def test_llm_former_packages_metadata() -> None:
    seg = strategies.Segment(
        turns=[_t("user", "do thing", 0), _t("agent", "did", 1)],
        topic_label="doing-thing",
        start_idx=0,
        end_idx=1,
        metadata={
            "worthwhile": True,
            "score": 8,
            "retrieval_key": "doing-thing-key",
            "summary": "did the thing successfully",
            "tags": ["thing"],
            "outcome": "success",
        },
    )
    former = strategies.LLMFormer()
    mem = former.form(seg, segment_idx=3)
    assert mem is not None
    assert mem.summary == "did the thing successfully"
    assert mem.retrieval_key == "doing-thing-key"
    assert mem.tags == ["thing"]
    assert mem.outcome == "success"
    assert mem.score == 8
    assert mem.formation_seconds == 0.0
    assert mem.source_segment_idx == 3


def test_llm_former_skips_unworthy_by_default() -> None:
    seg = strategies.Segment(
        turns=[_t("user", "ping", 0)],
        topic_label="ping",
        metadata={"worthwhile": False, "score": 1, "summary": "ping",
                  "retrieval_key": "ping", "tags": [], "outcome": None},
    )
    former = strategies.LLMFormer()
    assert former.form(seg) is None


def test_llm_former_include_unworthy_flag() -> None:
    seg = strategies.Segment(
        turns=[_t("user", "ping", 0)],
        topic_label="ping",
        metadata={"worthwhile": False, "score": 1, "summary": "ping",
                  "retrieval_key": "ping", "tags": [], "outcome": None},
    )
    former = strategies.LLMFormer(include_unworthy=True)
    mem = former.form(seg)
    assert mem is not None
    assert mem.summary == "ping"


def test_llm_former_returns_none_when_no_metadata() -> None:
    """LLMFormer paired with a non-LLM segmenter (no metadata) returns None.

    This prevents nonsense memories from being produced when wired wrong.
    """
    seg = strategies.Segment(
        turns=[_t("user", "x", 0)],
        topic_label="x",
        metadata={},
    )
    former = strategies.LLMFormer()
    assert former.form(seg) is None


# ──────────────────────────────────────────────────────────────────────
# strategies.py — formers
# ──────────────────────────────────────────────────────────────────────


def test_current_former_offline_returns_proposed_memory() -> None:
    former = strategies.CurrentFormer(offline=True)
    seg = strategies.Segment(
        turns=[_t("user", "build me a thing", 0), _t("agent", "ok done", 1)],
        topic_label="building",
        start_idx=0,
        end_idx=1,
    )
    mem = former.form(seg, segment_idx=7)
    assert mem is not None
    assert mem.topic_label == "building"
    assert mem.source_segment_idx == 7
    assert mem.source_turn_count == 2
    assert "[offline]" in mem.summary


def test_current_former_with_mocked_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    canned = (
        "<reflection>I did the thing and it worked.</reflection>\n"
        "<summary>Built a thing successfully; key lesson: read the docs first.</summary>\n"
        "<tags>building, docs</tags>\n"
        "<outcome_label>success</outcome_label>\n"
        "<retrieval_key>Built a thing for Project Owner using the standard pattern.</retrieval_key>\n"
        "<project>infra</project>\n"
    )
    monkeypatch.setattr(strategies, "_llm_chat", lambda *a, **k: canned)
    former = strategies.CurrentFormer(offline=False)
    seg = strategies.Segment(
        turns=[_t("user", "build me a thing", 0), _t("agent", "ok done", 1)],
        topic_label="building",
    )
    mem = former.form(seg)
    assert mem is not None
    assert mem.outcome == "success"
    assert mem.tags == ["building", "docs"]
    assert mem.project == "infra"
    assert "Built a thing successfully" in mem.summary
    assert mem.retrieval_key.startswith("Built a thing")


def test_current_former_empty_segment_returns_none() -> None:
    former = strategies.CurrentFormer(offline=True)
    assert former.form(strategies.Segment(turns=[])) is None


def test_batch_former_offline() -> None:
    former = strategies.BatchFormer(offline=True)
    seg = strategies.Segment(
        turns=[_t("user", "x", 0), _t("agent", "y", 1)],
        topic_label="batch-0",
    )
    mem = former.form(seg, segment_idx=2)
    assert mem is not None
    assert "[offline batch]" in mem.summary
    assert mem.source_segment_idx == 2


# ──────────────────────────────────────────────────────────────────────
# run.py — pipeline + schema
# ──────────────────────────────────────────────────────────────────────


def test_run_pipeline_offline_end_to_end() -> None:
    turns = [_t("user", f"q{i}", i) for i in range(6)]
    seg = strategies.BatchSegmenter(size=2)
    former = strategies.CurrentFormer(offline=True)
    out = run_pipeline(turns, segmenter=seg, former=former)
    assert out["input"]["turn_count"] == 6
    assert out["segmenter"]["name"] == "batch"
    assert out["former"]["name"] == "current"
    assert out["segments"]["count"] == 3
    assert out["proposed_memories"]["count"] == 3
    # schema invariants the comparison table relies on
    for key in ("turns_median", "single_turn_pct", "size_distribution"):
        assert key in out["segments"]
    for key in ("count", "outcomes", "error_summaries"):
        assert key in out["proposed_memories"]


def test_run_pipeline_no_former() -> None:
    turns = [_t("user", "a", i) for i in range(4)]
    seg = strategies.BatchSegmenter(size=2)
    out = run_pipeline(turns, segmenter=seg, former=None)
    assert out["proposed_memories"]["count"] == 0
    assert out["segments"]["count"] == 2


def test_segment_metrics_empty() -> None:
    assert segment_metrics([]) == {"count": 0}


def test_segment_metrics_distribution() -> None:
    segs = [
        strategies.Segment(turns=[_t("user", "a", 0)]),
        strategies.Segment(turns=[_t("user", "a", i) for i in range(3)]),
        strategies.Segment(turns=[_t("user", "a", i) for i in range(15)]),
    ]
    m = segment_metrics(segs)
    assert m["count"] == 3
    assert m["single_turn_pct"] == pytest.approx(33.3, abs=0.5)
    assert m["turns_max"] == 15
    assert m["size_distribution"][">=10"] == 1


def test_make_segmenter_unknown_raises() -> None:
    with pytest.raises(KeyError):
        strategies.make_segmenter("does-not-exist")


def test_make_former_unknown_raises() -> None:
    with pytest.raises(KeyError):
        strategies.make_former("does-not-exist")


def test_factories_cover_all_advertised_names() -> None:
    for name in strategies.SEGMENTERS:
        if name == "current":
            inst = strategies.make_segmenter(name, offline=True)
        else:
            inst = strategies.make_segmenter(name)
        assert isinstance(inst, strategies.Segmenter)
    for name in strategies.FORMERS:
        if name == "llm":
            inst = strategies.make_former(name)
        else:
            inst = strategies.make_former(name, offline=True)
        assert isinstance(inst, strategies.Former)
