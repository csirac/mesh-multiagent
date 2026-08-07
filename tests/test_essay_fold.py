"""Tests for the essay fold integration module (Phase 2).

Covers: admission/promotion logic (PATCH/CREATE/SKIP), entity recurrence
tracking, essay validation (empty, placeholder guard, token budget,
citation resolution), tool execution (essay_get/essay_list/essay_edit),
and prompt instruction formatting.
"""

import json
import os
import sqlite3
import tempfile

import pytest

import mesh.memory.essay_fold as essay_fold
from mesh.memory.essay_fold import (
    ESSAY_GENERATOR_SYSTEM,
    ESSAY_GENERATOR_USER_CREATE,
    ESSAY_GENERATOR_USER_PATCH,
    ESSAY_GENERATOR_USER_REGENERATE,
    ESSAYS_TABLE_DDL,
    EssayAction,
    BRACKET_TOKEN_RE,
    GENERATOR_TOOL_SCHEMAS,
    LOOSE_TAG_RE,
    MetaReviewItem,
    _detect_entity_collisions,
    _ensure_essays_table,
    _essay_type_from_key,
    _reset_dict_cache,
    _section_schema_for_type,
    build_seeding_prompt,
    check_admission,
    entity_window_count,
    exec_essay_tool,
    exec_generator_tool,
    extract_digest_entities,
    format_essay_instructions,
    format_meta_review_report,
    generate_essay,
    record_entity_mentions,
    scan_meta_review,
    validate_essay,
)


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_dict_cache():
    """Reset the module-level _DICT_WORDS cache between tests."""
    _reset_dict_cache()
    yield
    _reset_dict_cache()


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "test_unified.db")


@pytest.fixture
def db_with_essays(db_path):
    """DB with the essays table pre-created and one essay seeded."""
    con = sqlite3.connect(db_path)
    _ensure_essays_table(con)
    con.execute(
        "INSERT INTO essays (entity_key, title, body, citations, cross_refs, "
        "patch_count, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("person:kaylee", "Kaylee", "Kaylee is a key collaborator. [m_abc123def456]",
         '["m_abc123def456"]', '["project:mesh"]', 2,
         "2026-07-01T00:00:00+00:00", "2026-07-10T00:00:00+00:00"),
    )
    con.commit()
    con.close()
    return db_path


# ── Entity recurrence tracking ──────────────────────────────────

class TestRecurrenceTracking:
    def test_record_and_count(self, db_path):
        record_entity_mentions(db_path, ["person:alice", "project:mesh"], 1)
        assert entity_window_count(db_path, "person:alice") == 1
        assert entity_window_count(db_path, "project:mesh") == 1

    def test_same_round_no_double_count(self, db_path):
        record_entity_mentions(db_path, ["person:alice"], 1)
        record_entity_mentions(db_path, ["person:alice"], 1)
        assert entity_window_count(db_path, "person:alice") == 1

    def test_multiple_rounds(self, db_path):
        for r in range(5):
            record_entity_mentions(db_path, ["person:alice"], r)
        assert entity_window_count(db_path, "person:alice") == 5

    def test_unknown_entity_zero(self, db_path):
        record_entity_mentions(db_path, ["person:alice"], 1)
        assert entity_window_count(db_path, "person:bob") == 0


# ── Admission / promotion ──────────────────────────────────────

class TestAdmission:
    def test_skip_below_threshold(self, db_path):
        actions = check_admission(db_path, ["person:alice"], round_no=1, threshold=3)
        assert len(actions) == 1
        assert actions[0].action == EssayAction.SKIP
        assert actions[0].window_count == 1

    def test_create_at_threshold(self, db_path):
        for r in range(2):
            record_entity_mentions(db_path, ["person:alice"], r)
        actions = check_admission(db_path, ["person:alice"], round_no=2, threshold=3)
        assert actions[0].action == EssayAction.CREATE
        assert actions[0].window_count == 3

    def test_create_above_threshold(self, db_path):
        for r in range(4):
            record_entity_mentions(db_path, ["person:alice"], r)
        actions = check_admission(db_path, ["person:alice"], round_no=4, threshold=3)
        assert actions[0].action == EssayAction.CREATE
        assert actions[0].window_count == 5

    def test_patch_existing_essay(self, db_with_essays):
        actions = check_admission(
            db_with_essays, ["person:kaylee"], round_no=10, threshold=3)
        assert actions[0].action == EssayAction.PATCH
        assert actions[0].existing_body.startswith("Kaylee is a key")
        assert actions[0].existing_title == "Kaylee"

    def test_mixed_actions(self, db_with_essays):
        record_entity_mentions(db_with_essays, ["project:newproj"], 1)
        record_entity_mentions(db_with_essays, ["project:newproj"], 2)
        actions = check_admission(
            db_with_essays,
            ["person:kaylee", "project:newproj", "topic:weather"],
            round_no=3,
            threshold=3,
        )
        by_key = {a.entity_key: a for a in actions}
        assert by_key["person:kaylee"].action == EssayAction.PATCH
        assert by_key["project:newproj"].action == EssayAction.CREATE
        assert by_key["topic:weather"].action == EssayAction.SKIP

    def test_custom_threshold(self, db_path):
        record_entity_mentions(db_path, ["person:alice"], 1)
        actions = check_admission(db_path, ["person:alice"], round_no=2, threshold=2)
        assert actions[0].action == EssayAction.CREATE

    def test_threshold_one(self, db_path):
        actions = check_admission(db_path, ["topic:test"], round_no=0, threshold=1)
        assert actions[0].action == EssayAction.CREATE


# ── Validation ──────────────────────────────────────────────────

class TestValidation:
    def test_valid_essay(self):
        assert validate_essay("A valid essay body with [m_abc123def456].") is None

    def test_empty_body(self):
        assert validate_essay("") is not None
        assert "empty" in validate_essay("")

    def test_whitespace_only(self):
        assert validate_essay("   \n\t  ") is not None

    def test_placeholder_guard_double_bracket(self):
        err = validate_essay("Some text with [[placeholder]] remaining.")
        assert err is not None
        assert "double-bracket" in err
        assert "[[placeholder]]" in err

    def test_placeholder_guard_handle(self):
        err = validate_essay("Citing [[M3]] in the essay.")
        assert err is not None
        assert "[[M3]]" in err

    def test_multiple_placeholders(self):
        err = validate_essay("Text [[A]] and [[B]] here.")
        assert err is not None
        assert "[[A]]" in err
        assert "[[B]]" in err

    def test_canonical_citation_ok(self):
        assert validate_essay("Valid [m_abc123def456] citation.") is None

    def test_token_budget_under(self):
        assert validate_essay("short", token_budget=100, ntokens_fn=len) is None

    def test_token_budget_over(self):
        body = "x" * 200
        err = validate_essay(body, token_budget=100, ntokens_fn=len)
        assert err is not None
        assert "over budget" in err
        assert "100" in err

    def test_token_budget_exact(self):
        body = "x" * 100
        assert validate_essay(body, token_budget=100, ntokens_fn=len) is None

    def test_no_ntokens_fn_skips_budget(self):
        body = "x" * 10000
        assert validate_essay(body, token_budget=10) is None

    def test_citation_resolution_valid(self):
        known = {"abc123def456", "112233445566"}
        body = "Cites [m_abc123def456] and [m_112233445566]."
        assert validate_essay(body, known_ids=known) is None

    def test_citation_resolution_invalid(self):
        known = {"abc123def456"}
        body = "Cites [m_abc123def456] and [m_deadbeef1234]."
        err = validate_essay(body, known_ids=known)
        assert err is not None
        assert "deadbeef1234" in err

    def test_citation_resolution_no_known_ids_skips(self):
        body = "Cites [m_nonexistent123]."
        assert validate_essay(body, known_ids=None) is None


# ── Regex patterns ──────────────────────────────────────────────

class TestRegex:
    def test_bracket_token_matches(self):
        assert BRACKET_TOKEN_RE.findall("text [[M1]] more [[placeholder]]") == [
            "[[M1]]", "[[placeholder]]"]

    def test_bracket_token_no_match_single(self):
        assert BRACKET_TOKEN_RE.findall("[single]") == []

    def test_bracket_token_no_newline(self):
        assert BRACKET_TOKEN_RE.findall("[[multi\nline]]") == []

    def test_loose_tag_matches(self):
        ids = {m.group(1) for m in LOOSE_TAG_RE.finditer(
            "See [m_abc123] and m_def456.")}
        assert ids == {"abc123", "def456"}

    def test_loose_tag_no_word_prefix(self):
        ids = {m.group(1) for m in LOOSE_TAG_RE.finditer("from_dict()")}
        assert ids == set()

    def test_loose_tag_underscore_prefix(self):
        ids = {m.group(1) for m in LOOSE_TAG_RE.finditer("llm_backend")}
        assert ids == set()

    def test_loose_tag_uppercase_hex(self):
        ids = {m.group(1) for m in LOOSE_TAG_RE.finditer(
            "See [m_ABC123DEF456] and m_aaBBcc.")}
        assert ids == {"ABC123DEF456", "aaBBcc"}

    def test_citation_resolution_uppercase(self):
        known = {"ABC123DEF456"}
        body = "Cites [m_ABC123DEF456]."
        assert validate_essay(body, known_ids=known) is None


# ── DDL consistency ────────────────────────────────────────────

class TestDDLConsistency:
    def test_essay_fold_ddl_matches_store(self):
        """The essay_fold module's ESSAYS_TABLE_DDL must produce the same
        column set as mesh/memory/store.py's _create_tables() essays block."""
        fold_con = sqlite3.connect(":memory:")
        fold_con.execute(ESSAYS_TABLE_DDL)
        fold_cols = {
            (r[1], r[2]) for r in
            fold_con.execute("PRAGMA table_info(essays)").fetchall()
        }
        fold_con.close()

        from mesh.memory.store import MemoryStore
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            store = MemoryStore(os.path.join(d, "check.db"))
            store_cols = {
                (r[1], r[2]) for r in
                store._conn.execute("PRAGMA table_info(essays)").fetchall()
            }
            store._conn.close()

        assert fold_cols == store_cols, (
            f"Schema mismatch — essay_fold has {fold_cols}, "
            f"store.py has {store_cols}"
        )

    def test_ensure_essays_table_idempotent(self, db_path):
        con = sqlite3.connect(db_path)
        _ensure_essays_table(con)
        _ensure_essays_table(con)
        tables = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='essays'"
        ).fetchall()
        con.close()
        assert len(tables) == 1


# ── Tool execution ──────────────────────────────────────────────

class TestExecEssayTool:
    def test_essay_list_empty(self, db_path):
        result = exec_essay_tool("essay_list", {}, db_path)
        assert "No essays" in result

    def test_essay_create_and_get(self, db_path):
        result = exec_essay_tool("essay_edit", {
            "key": "person:alice",
            "old_text": "",
            "new_text": "Alice is a researcher. [m_aaa111bbb222]",
            "title": "Alice",
        }, db_path)
        assert "created" in result

        result = exec_essay_tool("essay_get", {"key": "person:alice"}, db_path)
        assert "Alice is a researcher" in result
        assert "m_aaa111bbb222" in result

    def test_essay_list_after_create(self, db_path):
        exec_essay_tool("essay_edit", {
            "key": "person:alice",
            "old_text": "", "new_text": "Body text.",
            "title": "Alice",
        }, db_path)
        result = exec_essay_tool("essay_list", {}, db_path)
        assert "1 essay" in result
        assert "person:alice" in result

    def test_essay_edit_existing(self, db_with_essays):
        result = exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "key collaborator",
            "new_text": "essential partner",
        }, db_with_essays)
        assert "updated" in result
        assert "1 replacement" in result

        result = exec_essay_tool("essay_get", {"key": "person:kaylee"}, db_with_essays)
        assert "essential partner" in result
        assert "key collaborator" not in result

    def test_essay_edit_old_text_not_found(self, db_with_essays):
        result = exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "nonexistent text",
            "new_text": "replacement",
        }, db_with_essays)
        assert "Error" in result
        assert "not found" in result

    def test_essay_edit_empty_old_text_refuses_existing(self, db_with_essays):
        result = exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "",
            "new_text": "Replacement should not be inserted between chars.",
            "replace_all": True,
        }, db_with_essays)
        assert "Error" in result
        assert "old_text is empty" in result

    def test_essay_edit_multiple_matches_no_replace_all(self, db_path):
        exec_essay_tool("essay_edit", {
            "key": "topic:test",
            "old_text": "", "new_text": "word word word",
        }, db_path)
        result = exec_essay_tool("essay_edit", {
            "key": "topic:test",
            "old_text": "word",
            "new_text": "term",
        }, db_path)
        assert "Error" in result
        assert "3 locations" in result

    def test_essay_edit_replace_all(self, db_path):
        exec_essay_tool("essay_edit", {
            "key": "topic:test",
            "old_text": "", "new_text": "word word word",
        }, db_path)
        result = exec_essay_tool("essay_edit", {
            "key": "topic:test",
            "old_text": "word",
            "new_text": "term",
            "replace_all": True,
        }, db_path)
        assert "3 replacement" in result
        body = exec_essay_tool("essay_get", {"key": "topic:test"}, db_path)
        assert "word" not in body
        assert "term" in body

    def test_essay_edit_with_citations(self, db_path):
        result = exec_essay_tool("essay_edit", {
            "key": "person:bob",
            "old_text": "", "new_text": "Bob's essay. [m_aaa111bbb222]",
            "title": "Bob",
            "citations": '["m_aaa111bbb222"]',
            "cross_refs": '["person:alice"]',
        }, db_path)
        assert "created" in result
        body = exec_essay_tool("essay_get", {"key": "person:bob"}, db_path)
        assert "m_aaa111bbb222" in body
        assert "person:alice" in body

    def test_essay_patch_merges_citation_metadata(self, db_path):
        exec_essay_tool("essay_edit", {
            "key": "person:bob",
            "old_text": "",
            "new_text": "First paragraph. [m_aaa111bbb222]",
            "title": "Bob",
            "citations": '["m_aaa111bbb222"]',
            "cross_refs": '["person:alice"]',
        }, db_path)
        result = exec_essay_tool("essay_edit", {
            "key": "person:bob",
            "old_text": "First paragraph.",
            "new_text": (
                "First paragraph.\n"
                "Second paragraph. [m_ccc333ddd444]"
            ),
            "citations": '["m_ccc333ddd444"]',
            "cross_refs": '["person:charlie"]',
        }, db_path)
        assert "updated" in result

        con = sqlite3.connect(db_path)
        body, citations, cross_refs = con.execute(
            "SELECT body, citations, cross_refs FROM essays WHERE entity_key = ?",
            ("person:bob",),
        ).fetchone()
        con.close()
        inline = [f"m_{m.group(1)}" for m in LOOSE_TAG_RE.finditer(body)]
        assert json.loads(citations) == inline
        assert set(json.loads(citations)) == {"m_aaa111bbb222", "m_ccc333ddd444"}
        assert json.loads(cross_refs) == ["person:alice", "person:charlie"]

    def test_essay_edit_normalizes_citation_metadata_to_inline_refs(self, db_path):
        exec_essay_tool("essay_edit", {
            "key": "person:bob",
            "old_text": "",
            "new_text": "First paragraph. [m_aaa111bbb222]",
            "title": "Bob",
            "citations": '["m_aaa111bbb222", "m_deadbeef0011"]',
        }, db_path)

        con = sqlite3.connect(db_path)
        citations = con.execute(
            "SELECT citations FROM essays WHERE entity_key = ?",
            ("person:bob",),
        ).fetchone()[0]
        con.close()
        assert json.loads(citations) == ["m_aaa111bbb222"]

        result = exec_essay_tool("essay_edit", {
            "key": "person:bob",
            "old_text": "First paragraph.",
            "new_text": (
                "First paragraph.\n"
                "Second paragraph. [m_ccc333ddd444]"
            ),
            "citations": '["m_deadbeef0011", "m_ccc333ddd444"]',
        }, db_path)
        assert "updated" in result

        con = sqlite3.connect(db_path)
        body, citations = con.execute(
            "SELECT body, citations FROM essays WHERE entity_key = ?",
            ("person:bob",),
        ).fetchone()
        con.close()
        inline = [f"m_{m.group(1)}" for m in LOOSE_TAG_RE.finditer(body)]
        assert json.loads(citations) == inline
        assert set(json.loads(citations)) == {"m_aaa111bbb222", "m_ccc333ddd444"}

    def test_essay_edit_invalid_citations_json(self, db_path):
        result = exec_essay_tool("essay_edit", {
            "key": "person:bob",
            "old_text": "", "new_text": "body",
            "citations": "not json",
        }, db_path)
        assert "Error" in result
        assert "citations JSON" in result

    def test_essay_get_missing(self, db_path):
        result = exec_essay_tool("essay_get", {"key": "person:nobody"}, db_path)
        assert "No essay found" in result

    def test_essay_get_no_key(self, db_path):
        result = exec_essay_tool("essay_get", {}, db_path)
        assert "Error" in result

    def test_unknown_tool(self, db_path):
        result = exec_essay_tool("essay_delete", {}, db_path)
        assert "Error" in result
        assert "unknown" in result

    def test_essay_edit_increments_patch_count(self, db_with_essays):
        exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "key collaborator",
            "new_text": "core contributor",
        }, db_with_essays)
        con = sqlite3.connect(db_with_essays)
        row = con.execute(
            "SELECT patch_count FROM essays WHERE entity_key = ?",
            ("person:kaylee",),
        ).fetchone()
        con.close()
        assert row[0] == 3  # was 2, now 3


# ── Prompt formatting ──────────────────────────────────────────

class TestFormatInstructions:
    def test_no_actions(self):
        result = format_essay_instructions([], essay_budget=4000)
        assert "No entities" in result

    def test_with_actions(self):
        actions = [
            EssayAction("person:alice", EssayAction.PATCH, window_count=5),
            EssayAction("project:mesh", EssayAction.CREATE, window_count=3),
        ]
        result = format_essay_instructions(actions, essay_budget=2000)
        assert "person:alice" in result
        assert "PATCH" in result
        assert "project:mesh" in result
        assert "CREATE" in result
        assert "2000" in result or "2,000" in result

    def test_budget_in_instructions(self):
        result = format_essay_instructions([], essay_budget=5000)
        assert "5000" in result

    def test_window_count_singular(self):
        actions = [EssayAction("topic:x", EssayAction.SKIP, window_count=1)]
        result = format_essay_instructions(actions)
        assert "1 window)" in result

    def test_window_count_plural(self):
        actions = [EssayAction("topic:x", EssayAction.PATCH, window_count=4)]
        result = format_essay_instructions(actions)
        assert "4 windows)" in result


# ── EssayAction repr ────────────────────────────────────────────

class TestEssayAction:
    def test_repr(self):
        a = EssayAction("person:alice", EssayAction.PATCH, window_count=3)
        assert "person:alice" in repr(a)
        assert "patch" in repr(a)
        assert "windows=3" in repr(a)

    def test_constants(self):
        assert EssayAction.PATCH == "patch"
        assert EssayAction.CREATE == "create"
        assert EssayAction.SKIP == "skip"


# ── Phase 3: Meta-review ──────────────────────────────────────

SAMPLE_DIGEST = """\
## Timeline

- **2026-04-01:** **Kaylee** conversation about honesty. [m_abc123def456]
- **2026-04-15:** **Nathan** discussed boundaries. [m_111222333444]
- **2026-05-01:** project:mesh-system refactor completed.
- **2026-05-10:** **Alice** reviewed the pipeline changes.
- **2026-06-01:** project:novelty-pipeline smoke test passed.

## People

**Kaylee** — key partner in self-work. [m_abc123def456] [m_aabbccdd1122]
**Nathan** — boundary discussions, hollowed-out dynamic. [m_111222333444]
"""


class TestExtractDigestEntities:
    def test_extracts_people(self):
        entities = extract_digest_entities(SAMPLE_DIGEST)
        assert "person:kaylee" in entities
        assert "person:nathan" in entities

    def test_extracts_projects(self):
        entities = extract_digest_entities(SAMPLE_DIGEST)
        assert "project:mesh-system" in entities
        assert "project:novelty-pipeline" in entities

    def test_skips_noise_words(self):
        entities = extract_digest_entities("**The** quick **Note** about **See**.")
        assert "person:the" not in entities
        assert "person:note" not in entities
        assert "person:see" not in entities

    def test_memory_refs_linked_to_nearby_entity(self):
        entities = extract_digest_entities(SAMPLE_DIGEST)
        assert "abc123def456" in entities.get("person:kaylee", [])

    def test_empty_digest(self):
        assert extract_digest_entities("") == {}

    def test_inline_single_name_extraction(self):
        digest = (
            "Kaylee said something. Then Kaylee replied. "
            "Later Kaylee clarified the point."
        )
        entities = extract_digest_entities(digest)
        assert "person:kaylee" in entities

    def test_inline_two_word_name_extraction(self):
        digest = (
            "Lee Ann cared deeply. Lee Ann showed up. "
            "Lee Ann was structurally incapable."
        )
        entities = extract_digest_entities(digest)
        assert "person:lee-ann" in entities

    def test_inline_below_threshold_excluded(self):
        digest = "Kaylee appeared once. Nathan appeared once."
        entities = extract_digest_entities(digest)
        assert "person:kaylee" not in entities
        assert "person:nathan" not in entities

    def test_noise_words_excluded_inline(self):
        digest = (
            "Pipeline ran. Pipeline ran. Pipeline ran again. "
            "Memory was checked. Memory was checked. Memory was checked. "
            "The code. The code. The code."
        )
        entities = extract_digest_entities(digest)
        assert "person:pipeline" not in entities
        assert "person:memory" not in entities
        assert "person:the" not in entities

    def test_bold_still_works_single_occurrence(self):
        digest = "One mention of **Kaylee** in bold."
        entities = extract_digest_entities(digest)
        assert "person:kaylee" in entities

    def test_noise_word_in_two_word_name_excluded(self):
        digest = (
            "Summer Day event. Summer Day event. Summer Day event."
        )
        entities = extract_digest_entities(digest)
        assert "person:summer-day" not in entities

    def test_coco_is_noise(self):
        digest = "Coco benchmark ran. Coco benchmark ran. Coco benchmark ran."
        entities = extract_digest_entities(digest)
        assert "person:coco" not in entities

    def test_alias_canonicalization_two_word(self):
        digest = (
            "Ankur Nath presented. Ankur Nath presented. Ankur Nath presented. "
            "Ankur helped. Ankur helped. Ankur helped. "
            "Nath reviewed. Nath reviewed. Nath reviewed. "
            "[m_aaa111] [m_bbb222] [m_ccc333]"
        )
        entities = extract_digest_entities(digest)
        assert "person:ankur-nath" in entities
        assert "person:ankur" not in entities, "standalone first-name should be folded"
        assert "person:nath" not in entities, "standalone surname should be folded"

    def test_alias_canonicalization_merges_refs(self):
        digest = (
            "Ankur helped with review. [m_aaa111] "
            "Nath reviewed the code. [m_bbb222] "
            "Ankur Nath presented results. Ankur Nath again. Ankur Nath again. "
            "Ankur worked. Ankur worked. Ankur worked. "
            "Nath coded. Nath coded. Nath coded. "
            "[m_ccc333]"
        )
        entities = extract_digest_entities(digest)
        assert "person:ankur-nath" in entities
        refs = entities["person:ankur-nath"]
        assert "aaa111" in refs or "bbb222" in refs or "ccc333" in refs

    def test_alias_canonicalization_preserves_non_overlapping(self):
        digest = (
            "Solmaz studies. Solmaz studies. Solmaz studies. "
            "Kaylee helped. Kaylee helped. Kaylee helped."
        )
        entities = extract_digest_entities(digest)
        assert "person:solmaz" in entities
        assert "person:kaylee" in entities

    def test_alias_canonicalization_creates_from_digest_mention(self):
        digest = (
            "Zorin Vesh reviewed the design. "
            "Zorin's code worked. Zorin's code worked. Zorin's code worked. "
            "Vesh published. Vesh published. Vesh published. "
            "[m_aaa111] [m_bbb222]"
        )
        entities = extract_digest_entities(digest)
        assert "person:zorin-vesh" in entities
        assert "person:zorin" not in entities
        assert "person:vesh" not in entities

    def test_alias_canonicalization_digest_merges_refs(self):
        digest = (
            "Zorin Vesh reviewed. "
            "Zorin's thing. Zorin's thing. Zorin's thing. [m_aaa111aaa111] "
            "Vesh wrote. Vesh wrote. Vesh wrote. [m_bbb222bbb222]"
        )
        entities = extract_digest_entities(digest)
        assert "person:zorin-vesh" in entities
        refs = entities["person:zorin-vesh"]
        assert "aaa111aaa111" in refs or "bbb222bbb222" in refs

    def test_alias_no_digest_merge_without_full_name(self):
        digest = (
            "Zorin's code. Zorin's code. Zorin's code. "
            "Vesh wrote. Vesh wrote. Vesh wrote."
        )
        entities = extract_digest_entities(digest)
        assert "person:zorin-vesh" not in entities
        assert "person:zorin" in entities
        assert "person:vesh" in entities

    def test_noise_words_block_db_corroboration_false_positives(self):
        for word in ["Amazon", "Architecture", "Benchmark", "Compiler",
                     "Investment", "Strategy", "Vanguard", "Session"]:
            digest = f"{word} appeared. {word} appeared. {word} appeared."
            entities = extract_digest_entities(digest)
            key = f"person:{word.lower()}"
            assert key not in entities, f"{word} should be in noise list"

    def test_db_corroboration_rescues_dict_word(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "test.db")
            con = sqlite3.connect(db)
            con.execute("""CREATE TABLE memories (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                summary TEXT NOT NULL, reflection TEXT NOT NULL,
                trace TEXT NOT NULL, trigger_text TEXT NOT NULL,
                retrieval_key TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL, outcome TEXT NOT NULL,
                reflection_embedding BLOB, retrieval_key_embedding BLOB,
                weight REAL NOT NULL DEFAULT 0.0,
                topic_label TEXT DEFAULT '', project TEXT DEFAULT '',
                digest_candidate INTEGER NOT NULL DEFAULT 1)""")
            for i in range(5):
                con.execute(
                    "INSERT INTO memories (id, created_at, summary, reflection, "
                    "trace, trigger_text, tags, outcome, topic_label) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (f"m_{i}", "2026-01-01", "s", "r", "t", "x", "", "success",
                     f"Nathan's evening discussion {i}"),
                )
            con.commit()
            con.close()

            digest = (
                "Nathan talked. Nathan talked. Nathan talked. "
                "Nathan again. [m_0]"
            )
            entities_no_db = extract_digest_entities(digest)
            assert "person:nathan" not in entities_no_db

            entities_with_db = extract_digest_entities(digest, db_path=db)
            assert "person:nathan" in entities_with_db

    def test_db_corroboration_no_db(self):
        digest = (
            "Nathan talked. Nathan talked. Nathan talked."
        )
        entities = extract_digest_entities(digest, db_path=None)
        assert "person:nathan" not in entities


class TestScanMetaReviewEmptyRefs:
    def test_seed_skips_empty_ref_entities(self, db_path):
        digest = (
            "Feldman worked on it. Feldman worked on it. Feldman worked on it."
        )
        items = scan_meta_review(db_path, digest)
        seed_keys = {i.entity_key for i in items
                     if i.action == MetaReviewItem.SEED}
        assert "person:feldman" not in seed_keys

    def test_seed_includes_entities_with_refs(self, db_path):
        items = scan_meta_review(db_path, SAMPLE_DIGEST)
        seed_keys = {i.entity_key for i in items
                     if i.action == MetaReviewItem.SEED}
        assert "person:kaylee" in seed_keys


class TestScanMetaReview:
    def test_seed_finding_for_missing_essay(self, db_path):
        items = scan_meta_review(db_path, SAMPLE_DIGEST)
        seed_items = [i for i in items if i.action == MetaReviewItem.SEED]
        seed_keys = {i.entity_key for i in seed_items}
        assert "person:kaylee" in seed_keys
        assert "person:nathan" in seed_keys

    def test_no_seed_for_existing_essay(self, db_with_essays):
        items = scan_meta_review(db_with_essays, SAMPLE_DIGEST)
        seed_keys = {i.entity_key for i in items
                     if i.action == MetaReviewItem.SEED}
        assert "person:kaylee" not in seed_keys

    def test_lumpy_finding(self, db_path):
        con = sqlite3.connect(db_path)
        _ensure_essays_table(con)
        con.execute(
            "INSERT INTO essays (entity_key, title, body, patch_count, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("person:kaylee", "Kaylee", "Essay body about **Kaylee**.",
             15, "2026-07-01T00:00:00+00:00", "2026-07-10T00:00:00+00:00"),
        )
        con.commit()
        con.close()
        items = scan_meta_review(db_path, SAMPLE_DIGEST, lumpy_threshold=10)
        lumpy = [i for i in items if i.action == MetaReviewItem.LUMPY]
        assert len(lumpy) == 1
        assert lumpy[0].entity_key == "person:kaylee"

    def test_stale_finding(self, db_path):
        con = sqlite3.connect(db_path)
        _ensure_essays_table(con)
        con.execute(
            "INSERT INTO essays (entity_key, title, body, patch_count, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("person:zebra", "Zebra", "Old essay.",
             0, "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"),
        )
        con.commit()
        con.close()
        items = scan_meta_review(db_path, SAMPLE_DIGEST)
        stale = [i for i in items if i.action == MetaReviewItem.STALE]
        stale_keys = {i.entity_key for i in stale}
        assert "person:zebra" in stale_keys

    def test_digest_tighten_finding(self, db_path):
        con = sqlite3.connect(db_path)
        _ensure_essays_table(con)
        con.execute(
            "INSERT INTO essays (entity_key, title, body, patch_count, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("person:kaylee", "Kaylee", "x" * 300,
             5, "2026-07-01T00:00:00+00:00", "2026-07-10T00:00:00+00:00"),
        )
        con.commit()
        con.close()
        items = scan_meta_review(db_path, SAMPLE_DIGEST)
        tighten = [i for i in items
                   if i.action == MetaReviewItem.DIGEST_TIGHTEN]
        assert any(i.entity_key == "person:kaylee" for i in tighten)

    def test_empty_digest_no_crash(self, db_path):
        items = scan_meta_review(db_path, "")
        assert items == []


class TestFormatMetaReviewReport:
    def test_no_findings(self):
        report = format_meta_review_report([])
        assert "no findings" in report

    def test_with_seed_findings(self):
        items = [
            MetaReviewItem("person:alice", MetaReviewItem.SEED,
                           "needs essay"),
            MetaReviewItem("person:bob", MetaReviewItem.LUMPY,
                           "too many patches"),
        ]
        report = format_meta_review_report(items)
        assert "2 finding" in report
        assert "person:alice" in report
        assert "Seeding" in report
        assert "Hygiene" in report

    def test_report_sorted_by_entity_key(self):
        items = [
            MetaReviewItem("person:zara", MetaReviewItem.SEED, "a"),
            MetaReviewItem("person:alice", MetaReviewItem.SEED, "b"),
        ]
        report = format_meta_review_report(items)
        assert report.index("person:alice") < report.index("person:zara")


class TestBuildSeedingPrompt:
    def test_includes_entity_key(self):
        prompt = build_seeding_prompt("person:kaylee", "digest text", [])
        assert "person:kaylee" in prompt

    def test_includes_memory_summaries(self):
        mems = [("abc123", "Had a conversation about honesty")]
        prompt = build_seeding_prompt("person:kaylee", "digest", mems)
        assert "m_abc123" in prompt
        assert "honesty" in prompt

    def test_no_memories(self):
        prompt = build_seeding_prompt("person:kaylee", "digest", [])
        assert "No raw memories" in prompt

    def test_budget_in_prompt(self):
        prompt = build_seeding_prompt("person:kaylee", "d", [], essay_budget=2000)
        assert "2000" in prompt

    def test_fold_complete_envelope(self):
        prompt = build_seeding_prompt("person:kaylee", "d", [])
        assert "<fold_complete>" in prompt
        assert "person:kaylee" in prompt


class TestMetaReviewItem:
    def test_repr(self):
        item = MetaReviewItem("person:alice", MetaReviewItem.SEED, "needs essay")
        assert "person:alice" in repr(item)
        assert "seed" in repr(item)

    def test_constants(self):
        assert MetaReviewItem.SEED == "seed"
        assert MetaReviewItem.LUMPY == "lumpy"
        assert MetaReviewItem.STALE == "stale"
        assert MetaReviewItem.DIGEST_TIGHTEN == "digest_tighten"
        assert MetaReviewItem.ENTITY_COLLISION == "entity_collision"


# ── Digest-gated admission ─────────────────────────────────────

DIGEST_GATED_SAMPLE = """\
## People

**Alice** has been instrumental in building the mesh framework. Alice's
contributions span the router and memory subsystems. [m_aaa111]

**Kaylee** continues to mentor new contributors and review PRs. [m_bbb222]

## Projects

The **mesh** project reached v2.0 this quarter with standing-digest support.

## Topics

Weather patterns remain a recurring theme in casual conversation.
"""


class TestDigestGatedAdmission:
    """Verify that check_admission() uses the digest as the authoritative gate
    when digest_text is provided."""

    def test_entity_in_digest_creates(self, db_path):
        """An entity present in the digest AND meeting recurrence threshold → CREATE."""
        for r in range(3):
            record_entity_mentions(db_path, ["person:alice"], r)
        actions = check_admission(
            db_path, ["person:alice"], round_no=3, threshold=3,
            digest_text=DIGEST_GATED_SAMPLE,
        )
        assert len(actions) == 1
        assert actions[0].action == EssayAction.CREATE
        assert actions[0].window_count >= 3

    def test_entity_not_in_digest_skips(self, db_path):
        """An entity NOT in the digest is SKIPPED even with high recurrence."""
        for r in range(10):
            record_entity_mentions(db_path, ["person:zara"], r)
        actions = check_admission(
            db_path, ["person:zara"], round_no=10, threshold=3,
            digest_text=DIGEST_GATED_SAMPLE,
        )
        assert len(actions) == 1
        assert actions[0].action == EssayAction.SKIP

    def test_digest_gate_still_requires_threshold(self, db_path):
        """In-digest but below recurrence threshold → SKIP (not CREATE)."""
        actions = check_admission(
            db_path, ["person:alice"], round_no=1, threshold=3,
            digest_text=DIGEST_GATED_SAMPLE,
        )
        assert len(actions) == 1
        assert actions[0].action == EssayAction.SKIP
        assert actions[0].window_count == 1

    def test_digest_gate_patches_existing(self, db_with_essays):
        """An entity in the digest with an existing essay → PATCH."""
        actions = check_admission(
            db_with_essays, ["person:kaylee"], round_no=10, threshold=3,
            digest_text=DIGEST_GATED_SAMPLE,
        )
        assert len(actions) == 1
        assert actions[0].action == EssayAction.PATCH

    def test_db_only_entity_never_admitted(self, db_path):
        """An entity only in the DB (not in digest) is never admitted,
        regardless of recurrence — the digest is the authoritative gate."""
        for r in range(20):
            record_entity_mentions(db_path, ["person:phantom"], r)
        actions = check_admission(
            db_path, ["person:phantom"], round_no=20, threshold=1,
            digest_text=DIGEST_GATED_SAMPLE,
        )
        assert len(actions) == 1
        assert actions[0].action == EssayAction.SKIP
        assert actions[0].window_count == 21

    def test_legacy_no_digest_still_works(self, db_path):
        """When digest_text is None, legacy recurrence-only gate applies."""
        for r in range(3):
            record_entity_mentions(db_path, ["person:phantom"], r)
        actions = check_admission(
            db_path, ["person:phantom"], round_no=3, threshold=3,
        )
        assert actions[0].action == EssayAction.CREATE

    def test_mixed_digest_and_non_digest(self, db_with_essays):
        """Multiple entities: some in digest, some not — correct per-entity behavior."""
        for r in range(4):
            record_entity_mentions(
                db_with_essays, ["person:alice", "person:nobody"], r)
        actions = check_admission(
            db_with_essays,
            ["person:kaylee", "person:alice", "person:nobody"],
            round_no=4, threshold=3,
            digest_text=DIGEST_GATED_SAMPLE,
        )
        by_key = {a.entity_key: a for a in actions}
        assert by_key["person:kaylee"].action == EssayAction.PATCH
        assert by_key["person:alice"].action == EssayAction.CREATE
        assert by_key["person:nobody"].action == EssayAction.SKIP


# ── Generator tool execution ─────────────────────────────────

@pytest.fixture
def db_with_memories(db_path):
    """DB with memories table, FTS index, and sample data."""
    con = sqlite3.connect(db_path)
    con.execute("""CREATE TABLE memories (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
        summary TEXT NOT NULL, reflection TEXT NOT NULL,
        trace TEXT NOT NULL, trigger_text TEXT NOT NULL,
        retrieval_key TEXT NOT NULL DEFAULT '',
        tags TEXT NOT NULL, outcome TEXT NOT NULL,
        reflection_embedding BLOB, retrieval_key_embedding BLOB,
        weight REAL NOT NULL DEFAULT 0.0,
        topic_label TEXT DEFAULT '', project TEXT DEFAULT '',
        digest_candidate INTEGER NOT NULL DEFAULT 1)""")
    con.execute("""CREATE VIRTUAL TABLE memories_fts USING fts5(
        id UNINDEXED, summary, reflection, retrieval_key)""")
    _ensure_essays_table(con)
    for i, (summ, refl, topic) in enumerate([
        ("Kaylee discussed honesty in relationships", "Deep vulnerability", "kaylee"),
        ("Kaylee shared childhood memory about trust", "Formative experience", "kaylee"),
        ("Pipeline architecture review with team", "Solid design", "pipeline"),
        ("Nathan boundary conversation evening", "Difficult dynamics", "nathan"),
        ("Kaylee's feedback on the fold system", "Practical and clear", "kaylee"),
    ]):
        mid = f"aaa{i:03d}bbb{i:03d}"
        con.execute(
            "INSERT INTO memories (id, created_at, summary, reflection, "
            "trace, trigger_text, tags, outcome, topic_label) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mid, f"2026-07-0{i+1}T12:00:00", summ, refl,
             "trace", "trigger", "", "success", topic),
        )
        con.execute(
            "INSERT INTO memories_fts (id, summary, reflection, retrieval_key) "
            "VALUES (?, ?, ?, ?)",
            (mid, summ, refl, ""),
        )
    con.commit()
    con.close()
    return db_path


def _create_collision_fixture(db_path, rows):
    con = sqlite3.connect(db_path)
    con.execute("""CREATE TABLE memories (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        summary TEXT NOT NULL,
        reflection TEXT NOT NULL DEFAULT '',
        retrieval_key TEXT NOT NULL DEFAULT '',
        topic_label TEXT DEFAULT '',
        project TEXT DEFAULT '',
        tags TEXT NOT NULL DEFAULT '')""")
    for i, row in enumerate(rows):
        con.execute(
            "INSERT INTO memories "
            "(id, created_at, summary, reflection, retrieval_key, "
            "topic_label, project, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"feed{i:03d}",
                f"2026-07-{i + 1:02d}T00:00:00+00:00",
                row.get("summary", ""),
                row.get("reflection", ""),
                row.get("retrieval_key", ""),
                row.get("topic_label", ""),
                row.get("project", ""),
                row.get("tags", ""),
            ),
        )
    con.commit()
    con.close()


class TestEntityCollisionDetection:
    def test_collision_detection_identical_entity(self, db_path):
        _create_collision_fixture(db_path, [
            {
                "summary": "Lily texted Project Owner about their Austin date.",
                "topic_label": "lily",
                "retrieval_key": "lily date follow-up",
            },
            {
                "summary": "Lily did not reply after the warm July text.",
                "topic_label": "lily",
                "retrieval_key": "lily july text",
            },
        ])

        assert _detect_entity_collisions(db_path, "person:lily") == []

    def test_collision_detection_distinct_entities(self, db_path):
        _create_collision_fixture(db_path, [
            {
                "summary": "Lily from Austin was the October date.",
                "topic_label": "lily-austin",
                "retrieval_key": "lily-austin dating thread",
            },
            {
                "summary": "Another Lily from Dallas works on the lab schedule.",
                "topic_label": "lily-work",
                "retrieval_key": "lily-work lab planning",
            },
        ])

        reasons = _detect_entity_collisions(db_path, "person:lily")
        assert reasons
        assert any("qualified topic labels" in reason for reason in reasons)

    def test_meta_review_reports_collision_warning(self, db_path):
        _create_collision_fixture(db_path, [
            {
                "summary": "Lily from Austin was the October date.",
                "topic_label": "lily-austin",
                "retrieval_key": "lily-austin dating thread",
            },
            {
                "summary": "Another Lily from Dallas works on the lab schedule.",
                "topic_label": "lily-work",
                "retrieval_key": "lily-work lab planning",
            },
        ])
        digest = "**Lily** has two active references. [m_feed000] [m_feed001]"

        items = scan_meta_review(db_path, digest)

        collisions = [
            item for item in items
            if item.action == MetaReviewItem.ENTITY_COLLISION
        ]
        assert collisions
        assert collisions[0].entity_key == "person:lily"


class TestExecGeneratorTool:
    def test_memory_search_fts(self, db_with_memories):
        result = exec_generator_tool("memory_search",
                                     {"query": "kaylee"}, db_with_memories)
        assert "result(s)" in result
        assert "kaylee" in result.lower()

    def test_memory_search_no_results(self, db_with_memories):
        result = exec_generator_tool("memory_search",
                                     {"query": "zzzznonexistent"}, db_with_memories)
        assert "No memories found" in result

    def test_memory_search_empty_query(self, db_with_memories):
        result = exec_generator_tool("memory_search",
                                     {"query": ""}, db_with_memories)
        assert "Error" in result

    def test_memory_get(self, db_with_memories):
        result = exec_generator_tool("memory_get",
                                     {"id": "aaa000bbb000"}, db_with_memories)
        assert "honesty" in result
        assert "m_aaa000bbb000" in result

    def test_memory_get_with_prefix(self, db_with_memories):
        result = exec_generator_tool("memory_get",
                                     {"id": "m_aaa000bbb000"}, db_with_memories)
        assert "honesty" in result

    def test_memory_get_not_found(self, db_with_memories):
        result = exec_generator_tool("memory_get",
                                     {"id": "nonexistent"}, db_with_memories)
        assert "No memory found" in result

    def test_memory_get_empty_id(self, db_with_memories):
        result = exec_generator_tool("memory_get", {"id": ""}, db_with_memories)
        assert "Error" in result

    def test_essay_edit_delegates(self, db_with_memories):
        result = exec_generator_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "",
            "new_text": "Test essay body.",
            "title": "Kaylee",
        }, db_with_memories)
        assert "created" in result

    def test_unknown_tool(self, db_with_memories):
        result = exec_generator_tool("unknown_tool", {}, db_with_memories)
        assert "Error" in result

    def test_memory_search_limit(self, db_with_memories):
        result = exec_generator_tool("memory_search",
                                     {"query": "kaylee", "limit": 1},
                                     db_with_memories)
        assert "1 result(s)" in result

    def test_memory_search_created_after_filter(self, db_with_memories):
        result = exec_generator_tool("memory_search", {
            "query": "kaylee",
            "created_after": "2026-07-04T00:00:00",
        }, db_with_memories)
        assert "fold system" in result
        assert "honesty" not in result
        assert "childhood memory" not in result

    def test_memory_search_entity_name_filter(self, db_with_memories):
        result = exec_generator_tool("memory_search", {
            "query": "kaylee",
            "entity_name": "nathan",
        }, db_with_memories)
        assert "No memories found" in result

        result = exec_generator_tool("memory_search", {
            "query": "kaylee",
            "entity_name": "kaylee",
        }, db_with_memories)
        assert "Kaylee discussed honesty" in result

    def test_memory_search_combined_filters(self, db_with_memories):
        result = exec_generator_tool("memory_search", {
            "query": "kaylee",
            "entity_name": "kaylee",
            "created_after": "2026-07-04T00:00:00",
        }, db_with_memories)
        assert "fold system" in result
        assert "honesty" not in result

    def test_memory_search_filters_can_exclude_everything(self, db_with_memories):
        result = exec_generator_tool("memory_search", {
            "query": "kaylee",
            "entity_name": "kaylee",
            "created_after": "2026-07-06T00:00:00",
        }, db_with_memories)
        assert "No memories found" in result

    def test_memory_search_schema_has_structured_filters(self):
        schema = next(
            t for t in GENERATOR_TOOL_SCHEMAS
            if t["function"]["name"] == "memory_search"
        )
        props = schema["function"]["parameters"]["properties"]
        assert "created_after" in props
        assert "entity_name" in props
        assert schema["function"]["parameters"]["required"] == ["query"]


# ── Generator ReAct loop ─────────────────────────────────────

def _make_tool_call(name, args, call_id="tc_001"):
    return [{"id": call_id, "function": {
        "name": name, "arguments": json.dumps(args)}}]


class TestGenerateEssay:
    def test_create_success(self, db_with_memories):
        """LLM searches, writes, and completes — happy path."""
        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("Searching for kaylee memories.",
                        _make_tool_call("memory_search",
                                        {"query": "kaylee"}, "tc_1"), {})
            if call_count[0] == 2:
                return ("Reading a memory.",
                        _make_tool_call("memory_get",
                                        {"id": "aaa000bbb000"}, "tc_2"), {})
            if call_count[0] == 3:
                return ("Writing the essay.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "",
                            "new_text": "Kaylee is central. [m_aaa000bbb000]",
                            "title": "Kaylee",
                            "citations": '["m_aaa000bbb000"]',
                        }, "tc_3"), {})
            return (
                "Done. <fold_complete><covering_from>generate"
                "</covering_from><covering_to>person:kaylee"
                "</covering_to></fold_complete>",
                [], {},
            )

        ok, msg = generate_essay(
            "person:kaylee", "digest about Kaylee", db_with_memories,
            fake_llm, essay_budget=4000,
        )
        assert ok
        assert "4 steps" in msg

        result = exec_essay_tool("essay_get",
                                 {"key": "person:kaylee"}, db_with_memories)
        assert "Kaylee is central" in result

    def test_validation_failure_retries(self, db_with_memories):
        """If essay fails validation, LLM gets feedback and retries."""
        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("Writing.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "",
                            "new_text": "Bad [[placeholder]] essay.",
                            "title": "Kaylee",
                        }, "tc_1"), {})
            if call_count[0] == 2:
                return (
                    "<fold_complete><covering_from>generate</covering_from>"
                    "<covering_to>person:kaylee</covering_to></fold_complete>",
                    [], {},
                )
            if call_count[0] == 3:
                return ("Fixing.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "Bad [[placeholder]] essay.",
                            "new_text": "Fixed essay body. [m_aaa000bbb000]",
                        }, "tc_3"), {})
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, msg = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, essay_budget=4000,
        )
        assert ok
        result = exec_essay_tool("essay_get",
                                 {"key": "person:kaylee"}, db_with_memories)
        assert "[[placeholder]]" not in result
        assert "Fixed essay" in result

    def test_complete_without_essay_retries(self, db_with_memories):
        """If LLM emits fold_complete without writing, it gets feedback."""
        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    "<fold_complete><covering_from>generate</covering_from>"
                    "<covering_to>person:kaylee</covering_to></fold_complete>",
                    [], {},
                )
            if call_count[0] == 2:
                return ("Writing.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "",
                            "new_text": "Essay body.",
                            "title": "Kaylee",
                        }, "tc_2"), {})
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, essay_budget=4000,
        )
        assert ok

    def test_max_rounds_exhausted(self, db_with_memories):
        """If LLM never completes, generate_essay returns failure."""
        def fake_llm(messages, tools):
            return ("Still thinking...", [], {})

        ok, msg = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, max_rounds=3,
        )
        assert not ok
        assert "3 rounds" in msg

    def test_tool_logging_includes_arguments(self, db_with_memories):
        """Generator logs include full tool arguments for smoke auditability."""
        logs = []
        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("Searching.",
                        _make_tool_call("memory_search", {
                            "query": "kaylee",
                            "limit": 1,
                        }, "tc_1"), {})
            if call_count[0] == 2:
                return ("Writing.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "",
                            "new_text": "Essay body [m_aaa000bbb000].",
                            "title": "Kaylee",
                        }, "tc_2"), {})
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, known_ids={"aaa000bbb000"}, log_fn=logs.append,
        )

        assert ok
        assert any(
            'memory_search({"query": "kaylee", "limit": 1}) OK' in log
            for log in logs
        )

    def test_patch_action(self, db_with_memories):
        """Patch action uses PATCH prompt template."""
        exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "",
            "new_text": "Original essay about Kaylee.",
            "title": "Kaylee",
        }, db_with_memories)

        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            assert "update existing essay" in messages[1]["content"].lower() \
                   or "existing essay" in messages[1]["content"].lower()
            if call_count[0] == 1:
                return ("Updating.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "Original essay about Kaylee.",
                            "new_text": "Updated essay about Kaylee. [m_aaa000bbb000]",
                        }, "tc_1"), {})
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, action="patch",
            existing_body="Original essay about Kaylee.",
            existing_title="Kaylee",
        )
        assert ok

    def test_create_action_auto_patches_existing_essay(self, db_with_memories):
        """CREATE requests against existing essays are safely routed to PATCH."""
        exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "",
            "new_text": "Original essay about Kaylee.",
            "title": "Kaylee",
        }, db_with_memories)

        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            assert "update existing essay" in messages[1]["content"].lower() \
                   or "existing essay" in messages[1]["content"].lower()
            if call_count[0] == 1:
                return ("Updating.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "Original essay about Kaylee.",
                            "new_text": "Original essay about Kaylee.\n"
                                        "New update. [m_aaa000bbb000]",
                        }, "tc_1"), {})
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, action="create",
        )
        assert ok

        result = exec_essay_tool("essay_get",
                                 {"key": "person:kaylee"}, db_with_memories)
        assert "New update" in result

    def test_regenerate_action(self, db_with_memories):
        """Regenerate action uses REGENERATE prompt template."""
        captured_prompts = []

        def fake_llm(messages, tools):
            captured_prompts.append(messages[1]["content"])
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                _make_tool_call("essay_edit", {
                    "key": "person:kaylee",
                    "old_text": "",
                    "new_text": "Regenerated. [m_aaa000bbb000]",
                    "title": "Kaylee",
                }, "tc_1"),
                {},
            )

        # Need an existing essay for the complete signal to validate against
        exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "", "new_text": "Old patchy essay.",
            "title": "Kaylee",
        }, db_with_memories)

        # The regenerate action will write a new essay via essay_edit, but
        # since the LLM returns both tool_calls and content in the same turn,
        # the tool calls are processed first, then the next turn will see the
        # completion. Let me restructure the fake.
        call_count = [0]

        def fake_llm2(messages, tools):
            call_count[0] += 1
            captured_prompts.append(messages[1]["content"])
            if call_count[0] == 1:
                return ("Regenerating.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "Old patchy essay.",
                            "new_text": "Fresh narrative. [m_aaa000bbb000]",
                            "replace_all": True,
                        }, "tc_1"), {})
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm2, action="regenerate",
            existing_body="Old patchy essay.",
            existing_title="Kaylee",
        )
        assert ok
        assert any("regenerat" in p.lower() for p in captured_prompts)

    def test_log_fn_called(self, db_with_memories):
        """log_fn receives progress messages."""
        logs = []

        def fake_llm(messages, tools):
            return ("Searching.",
                    _make_tool_call("memory_search",
                                    {"query": "kaylee"}, "tc_1"), {})

        generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, max_rounds=2, log_fn=logs.append,
        )
        assert any("memory_search" in l for l in logs)

    def test_malformed_tool_args(self, db_with_memories):
        """Malformed JSON in tool arguments doesn't crash the loop."""
        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("Bad call.", [{"id": "tc_1", "function": {
                    "name": "memory_search",
                    "arguments": "not json{{"}}], {})
            return ("Giving up.", [], {})

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, max_rounds=3,
        )
        assert not ok

    def test_generator_tool_schemas_structure(self):
        """GENERATOR_TOOL_SCHEMAS has the expected tools."""
        names = {t["function"]["name"] for t in GENERATOR_TOOL_SCHEMAS}
        assert names == {"memory_search", "memory_get", "essay_edit"}

    def test_known_ids_validation(self, db_with_memories):
        """Citation validation against known_ids works."""
        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("Writing.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "",
                            "new_text": "Essay citing [m_deadbeef1234].",
                            "title": "Kaylee",
                        }, "tc_1"), {})
            if call_count[0] == 2:
                return (
                    "<fold_complete><covering_from>generate</covering_from>"
                    "<covering_to>person:kaylee</covering_to></fold_complete>",
                    [], {},
                )
            # After validation failure, fix the citation
            if call_count[0] == 3:
                return ("Fixing.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "Essay citing [m_deadbeef1234].",
                            "new_text": "Essay citing [m_aaa000bbb000].",
                            "replace_all": True,
                        }, "tc_3"), {})
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, known_ids={"aaa000bbb000"},
        )
        assert ok


# ── Type detection and section schemas ────────────────────────────

class TestEssayTypeDetection:
    def test_person_default(self):
        assert _essay_type_from_key("person:kaylee") == "person"

    def test_person_bare_key(self):
        assert _essay_type_from_key("kaylee") == "person"

    def test_project_prefix(self):
        assert _essay_type_from_key("project:mesh") == "project"

    def test_event_prefix(self):
        assert _essay_type_from_key("event:july-4-bbq") == "event"


class TestSectionSchemas:
    def test_person_has_five_sections(self):
        schema = _section_schema_for_type("person")
        for heading in ["Identity", "Timeline", "Narrative",
                        "Reflection", "Open threads"]:
            assert heading in schema

    def test_person_no_related_section(self):
        schema = _section_schema_for_type("person")
        assert 'Do NOT create a separate "Related" section' in schema

    def test_person_facts_vs_interpretation(self):
        schema = _section_schema_for_type("person")
        assert "Facts-vs-interpretation boundary" in schema
        assert "cited fact ONLY" in schema

    def test_project_has_related_section(self):
        schema = _section_schema_for_type("project")
        assert "**Related**" in schema

    def test_project_sections(self):
        schema = _section_schema_for_type("project")
        for heading in ["Goal", "Decision arc", "Current state",
                        "Open issues", "Related"]:
            assert heading in schema

    def test_event_sections(self):
        schema = _section_schema_for_type("event")
        for heading in ["What happened", "Significance", "Aftermath"]:
            assert heading in schema

    def test_event_no_related_section(self):
        schema = _section_schema_for_type("event")
        assert "no separate Related section" in schema

    def test_cross_refs_in_all_types(self):
        for t in ["person", "project", "event"]:
            schema = _section_schema_for_type(t)
            assert "cross_refs" in schema


# ── Prompt template integration ───────────────────────────────────

class TestPromptTemplates:
    def test_system_prompt_contains_section_schema_placeholder(self):
        assert "{section_schema}" in ESSAY_GENERATOR_SYSTEM

    def test_system_prompt_contains_entity_type(self):
        assert "{entity_type}" in ESSAY_GENERATOR_SYSTEM

    def test_system_prompt_no_thematic_sections(self):
        assert "thematic sections" not in ESSAY_GENERATOR_SYSTEM

    def test_system_prompt_no_mini_digest(self):
        assert "mini-digest" not in ESSAY_GENERATOR_SYSTEM

    def test_system_prompt_renders_person(self):
        rendered = ESSAY_GENERATOR_SYSTEM.format(
            agent="alice", budget=4000, entity_key="person:kaylee",
            entity_type="person",
            section_schema=_section_schema_for_type("person"),
        )
        assert "Identity" in rendered
        assert "Timeline" in rendered
        assert "person" in rendered

    def test_system_prompt_renders_project(self):
        rendered = ESSAY_GENERATOR_SYSTEM.format(
            agent="alice", budget=4000, entity_key="project:mesh",
            entity_type="project",
            section_schema=_section_schema_for_type("project"),
        )
        assert "Decision arc" in rendered
        assert "**Related**" in rendered

    def test_create_template_has_entity_type(self):
        assert "{entity_type}" in ESSAY_GENERATOR_USER_CREATE

    def test_create_prompt_includes_disambiguation_guidance(self):
        lowered = ESSAY_GENERATOR_USER_CREATE.lower()
        assert "possible entity collision" in lowered
        assert "consider disambiguation" in lowered

    def test_patch_template_has_entity_type(self):
        assert "{entity_type}" in ESSAY_GENERATOR_USER_PATCH

    def test_patch_prompt_does_not_need_disambiguation_guidance(self):
        assert "possible entity collision" not in ESSAY_GENERATOR_USER_PATCH.lower()

    def test_patch_template_has_recency_placeholders(self):
        assert "{essay_updated_at}" in ESSAY_GENERATOR_USER_PATCH
        assert "{entity_name}" in ESSAY_GENERATOR_USER_PATCH

    def test_patch_template_has_prefetched_memories_placeholder(self):
        assert "{prefetched_memories}" in ESSAY_GENERATOR_USER_PATCH

    def test_create_and_regenerate_templates_do_not_prefetch(self):
        assert "{prefetched_memories}" not in ESSAY_GENERATOR_USER_CREATE
        assert "{prefetched_memories}" not in ESSAY_GENERATOR_USER_REGENERATE

    def test_patch_template_instructs_proactive_search(self):
        assert "Search for memories about" in ESSAY_GENERATOR_USER_PATCH
        assert "created or updated after" in ESSAY_GENERATOR_USER_PATCH
        assert "digest may not mention" in ESSAY_GENERATOR_USER_PATCH

    def test_patch_template_no_passive_digest_scan_language(self):
        lowered = ESSAY_GENERATOR_USER_PATCH.lower()
        assert "scan the digest" not in lowered
        assert "what's new" not in lowered
        assert "what is new" not in lowered

    def test_regenerate_template_has_entity_type(self):
        assert "{entity_type}" in ESSAY_GENERATOR_USER_REGENERATE


# ── known_ids auto-derivation ─────────────────────────────────────

class TestKnownIdsDerivation:
    def test_known_ids_derived_from_db(self, db_with_memories):
        """When known_ids=None, generate_essay derives them from the DB."""
        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("Writing.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "",
                            "new_text": "Essay about Kaylee [m_aaa000bbb000].",
                            "title": "Kaylee",
                        }, "tc_1"), {})
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, known_ids=None,
        )
        assert ok

    def test_known_ids_derived_rejects_unknown(self, db_with_memories):
        """Auto-derived known_ids still rejects citations not in the DB."""
        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("Writing.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "",
                            "new_text": "Essay [m_fff999fff999].",
                            "title": "Kaylee",
                        }, "tc_1"), {})
            if call_count[0] == 2:
                return (
                    "<fold_complete><covering_from>generate</covering_from>"
                    "<covering_to>person:kaylee</covering_to></fold_complete>",
                    [], {},
                )
            if call_count[0] == 3:
                return ("Fixing.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "Essay [m_fff999fff999].",
                            "new_text": "Essay [m_aaa000bbb000].",
                            "replace_all": True,
                        }, "tc_3"), {})
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, known_ids=None,
        )
        assert ok
        assert call_count[0] >= 3  # had to fix citation


# ── Multiple completion envelopes ─────────────────────────────────

class TestMultipleCompletionEnvelopes:
    def test_multiple_envelopes_takes_last(self, db_with_memories):
        """When LLM emits multiple fold_complete envelopes, use the last."""
        call_count = [0]

        def fake_llm(messages, tools):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("Writing.",
                        _make_tool_call("essay_edit", {
                            "key": "person:kaylee",
                            "old_text": "",
                            "new_text": "Kaylee essay [m_aaa000bbb000].",
                            "title": "Kaylee",
                        }, "tc_1"), {})
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:wrong</covering_to></fold_complete>\n"
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, known_ids={"aaa000bbb000"},
        )
        assert ok


# ── generate_essay type-awareness ─────────────────────────────────

class TestGenerateEssayTypeAwareness:
    def test_person_prompt_has_identity_section(self, db_with_memories):
        """generate_essay for a person entity injects person section schema."""
        captured_messages = []

        def fake_llm(messages, tools):
            captured_messages.extend(messages)
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                _make_tool_call("essay_edit", {
                    "key": "person:kaylee",
                    "old_text": "",
                    "new_text": "Essay [m_aaa000bbb000].",
                    "title": "Kaylee",
                }, "tc_1"), {},
            )

        generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, known_ids={"aaa000bbb000"},
        )
        system = captured_messages[0]["content"]
        assert "Identity" in system
        assert "Timeline" in system
        assert "person" in system

    def test_project_prompt_has_decision_arc(self, db_with_memories):
        """generate_essay for a project entity injects project schema."""
        captured_messages = []

        def fake_llm(messages, tools):
            captured_messages.extend(messages)
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>project:mesh</covering_to></fold_complete>",
                _make_tool_call("essay_edit", {
                    "key": "project:mesh",
                    "old_text": "",
                    "new_text": "Mesh essay [m_aaa000bbb000].",
                    "title": "Mesh",
                }, "tc_1"), {},
            )

        generate_essay(
            "project:mesh", "digest", db_with_memories,
            fake_llm, known_ids={"aaa000bbb000"},
        )
        system = captured_messages[0]["content"]
        assert "Decision arc" in system
        assert "project" in system
        user = captured_messages[1]["content"]
        assert "(type: project)" in user

    def test_patch_prompt_receives_updated_at_and_entity_name(
        self, db_with_memories,
    ):
        """PATCH prompt carries the essay timestamp and searchable entity name."""
        exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "",
            "new_text": "Existing essay [m_aaa000bbb000].",
            "title": "Kaylee",
        }, db_with_memories)
        con = sqlite3.connect(db_with_memories)
        con.execute(
            "UPDATE essays SET updated_at = ? WHERE entity_key = ?",
            ("2026-07-13T16:00:00+00:00", "person:kaylee"),
        )
        con.commit()
        con.close()

        captured_messages = []

        def fake_llm(messages, tools):
            captured_messages.extend(messages)
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, action="patch", known_ids={"aaa000bbb000"},
        )

        assert ok
        user = captured_messages[1]["content"]
        assert "2026-07-13T16:00:00+00:00" in user
        assert "Search for memories about **kaylee**" in user
        assert "digest may not mention" in user

    def test_patch_prompt_includes_prefetched_recent_memories(
        self, db_with_memories,
    ):
        """PATCH prompt gets deterministic recent entity memories before ReAct."""
        exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "",
            "new_text": "Existing essay [m_aaa000bbb000].",
            "title": "Kaylee",
        }, db_with_memories)
        con = sqlite3.connect(db_with_memories)
        con.execute(
            "UPDATE essays SET updated_at = ? WHERE entity_key = ?",
            ("2026-07-03T00:00:00", "person:kaylee"),
        )
        con.commit()
        con.close()

        captured_messages = []

        def fake_llm(messages, tools):
            captured_messages.extend(messages)
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, action="patch", known_ids={"aaa000bbb000"},
        )

        assert ok
        user = captured_messages[1]["content"]
        assert "## New Material to Review" in user
        assert "aaa004bbb004" in user
        assert "Kaylee's feedback on the fold system" in user
        assert "Kaylee discussed honesty" not in user

    def test_patch_prompt_reports_when_no_prefetched_memories(
        self, db_with_memories,
    ):
        """PATCH prompt explicitly says when no new entity memories were found."""
        exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "",
            "new_text": "Existing essay [m_aaa000bbb000].",
            "title": "Kaylee",
        }, db_with_memories)
        con = sqlite3.connect(db_with_memories)
        con.execute(
            "UPDATE essays SET updated_at = ? WHERE entity_key = ?",
            ("2026-07-06T00:00:00", "person:kaylee"),
        )
        con.commit()
        con.close()

        captured_messages = []

        def fake_llm(messages, tools):
            captured_messages.extend(messages)
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, action="patch", known_ids={"aaa000bbb000"},
        )

        assert ok
        user = captured_messages[1]["content"]
        assert "## New Material to Review" in user
        assert "No new memories found since the essay was last updated." in user

    def test_patch_prefetch_uses_entity_and_timestamp_filters(
        self, db_with_memories, monkeypatch,
    ):
        """Prefetch calls memory_search with entity_name and created_after."""
        exec_essay_tool("essay_edit", {
            "key": "person:kaylee",
            "old_text": "",
            "new_text": "Existing essay [m_aaa000bbb000].",
            "title": "Kaylee",
        }, db_with_memories)
        con = sqlite3.connect(db_with_memories)
        con.execute(
            "UPDATE essays SET updated_at = ? WHERE entity_key = ?",
            ("2026-07-13T16:00:00+00:00", "person:kaylee"),
        )
        con.commit()
        con.close()

        calls = []

        def fake_search(db_path, query, limit=20,
                        created_after=None, entity_name=None):
            calls.append({
                "db_path": db_path,
                "query": query,
                "limit": limit,
                "created_after": created_after,
                "entity_name": entity_name,
            })
            return "No memories found for query ''."

        monkeypatch.setattr(essay_fold, "_exec_memory_search", fake_search)

        def fake_llm(messages, tools):
            return (
                "<fold_complete><covering_from>generate</covering_from>"
                "<covering_to>person:kaylee</covering_to></fold_complete>",
                [], {},
            )

        ok, _ = generate_essay(
            "person:kaylee", "digest", db_with_memories,
            fake_llm, action="patch", known_ids={"aaa000bbb000"},
        )

        assert ok
        assert calls == [{
            "db_path": db_with_memories,
            "query": "",
            "limit": 50,
            "created_after": "2026-07-13T16:00:00+00:00",
            "entity_name": "kaylee",
        }]
