"""Validation tests for the optional Project Model nested in dossier Narrative."""

from pathlib import Path

import pytest

from mesh import project_dossier as pd


KEY = "project:project-model-test"
NARRATIVE = (
    "This dossier was initialized from the autonomous-agent-mode skeleton. No\n"
    "interpretive history has accrued yet."
)
VALID_BLOCK = """### Project Model

#### Observations

- `[verified]` `2026-08-05` `T-013, mesh/project_dossier.py:342` — The dossier validator enforces a bounded Project Model inside Narrative.

#### Current approach

- `[inferred]` `2026-08-05` `G-005, T-013` — Keep the model nested in Narrative while the seven-section constitution remains fixed. Reconsider if: the evaluation gate finds a stale-override failure.
"""


@pytest.fixture
def dossier(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "DIGESTS_DIR", tmp_path / "digests")
    return pd.init_dossier(KEY, owner_agent="coder1")


def _with_project_model(block: str = VALID_BLOCK) -> str:
    return pd.render_skeleton(KEY, owner_agent="coder1").replace(
        NARRATIVE,
        f"{NARRATIVE}\n\n{block.rstrip()}",
    )


def _over_budget_block() -> str:
    statement = "verified material finding " * 1800
    return VALID_BLOCK.replace(
        "The dossier validator enforces a bounded Project Model inside Narrative.",
        statement.strip() + ".",
    )


def test_valid_project_model_block_passes():
    assert pd._validate_project_model_block(VALID_BLOCK) == []


def test_absent_project_model_block_is_optional():
    assert pd._validate_project_model_block("Narrative without a model block.") == []


def test_valid_project_model_passes_full_dossier_validation():
    assert pd.validate_dossier(KEY, _with_project_model()) == []


def test_project_model_over_1200_tokens_is_refused():
    errors = pd._validate_project_model_block(_over_budget_block())
    assert any(
        error.startswith("Project Model block is over budget:")
        and "tokens > 1200 ceiling" in error
        for error in errors
    )


def test_project_model_missing_required_subheading_is_refused():
    block = VALID_BLOCK.replace("#### Observations\n", "")
    assert pd._validate_project_model_block(block) == [
        "Project Model block is missing required sub-section: Observations"
    ]


def test_project_model_empty_subsection_is_refused():
    observation = (
        "- `[verified]` `2026-08-05` `T-013, mesh/project_dossier.py:342` — "
        "The dossier validator enforces a bounded Project Model inside Narrative.\n"
    )
    block = VALID_BLOCK.replace(observation, "")
    assert "Project Model sub-section is empty: Observations" in (
        pd._validate_project_model_block(block)
    )


def test_project_model_malformed_epistemic_state_is_refused():
    block = VALID_BLOCK.replace("`[verified]`", "`[possible]`", 1)
    errors = pd._validate_project_model_block(block)
    assert any(
        "invalid epistemic state 'possible'" in error
        and "expected verified/inferred/uncertain/superseded" in error
        for error in errors
    )


def test_project_model_invalid_timestamp_is_refused():
    block = VALID_BLOCK.replace("`2026-08-05`", "`2026-8-5`", 1)
    assert any(
        "invalid timestamp '2026-8-5' (expected YYYY-MM-DD)" in error
        for error in pd._validate_project_model_block(block)
    )


def test_project_model_missing_evidence_pointer_is_refused():
    block = VALID_BLOCK.replace(
        "`T-013, mesh/project_dossier.py:342`",
        "``",
        1,
    )
    assert any(
        "missing evidence pointer" in error
        for error in pd._validate_project_model_block(block)
    )


def test_project_model_unrecognized_line_is_refused():
    block = VALID_BLOCK.replace(
        "#### Observations\n",
        "#### Observations\n\nThis prose is not a schema item.\n",
    )
    assert any(
        "unrecognized format" in error
        for error in pd._validate_project_model_block(block)
    )


def test_project_model_schema_applies_outside_required_subsections():
    block = VALID_BLOCK.replace(
        "#### Observations\n",
        "- `[possible]` `not-a-date` `` — Stray malformed item.\n\n"
        "#### Observations\n",
    )
    errors = pd._validate_project_model_block(block)
    assert any("invalid epistemic state 'possible'" in error for error in errors)
    assert any("invalid timestamp 'not-a-date'" in error for error in errors)
    assert any("missing evidence pointer" in error for error in errors)


def test_superseded_item_must_name_replacing_evidence():
    block = VALID_BLOCK.replace("`[verified]`", "`[superseded]`", 1)
    assert any(
        "superseded item must name what superseded it" in error
        for error in pd._validate_project_model_block(block)
    )


def test_current_approach_requires_final_reconsideration_trigger():
    block = VALID_BLOCK.replace(
        " Reconsider if: the evaluation gate finds a stale-override failure.",
        "",
    )
    assert any(
        "missing a final reconsideration trigger" in error
        for error in pd._validate_project_model_block(block)
    )


def test_edit_refuses_over_budget_project_model_and_preserves_bytes(dossier):
    before = dossier.read_text()
    with pytest.raises(pd.DossierError, match="Project Model block is over budget"):
        pd.edit_dossier(
            KEY,
            NARRATIVE,
            f"{NARRATIVE}\n\n{_over_budget_block().rstrip()}",
        )
    assert dossier.read_bytes() == before.encode()


def test_plan_prompt_reads_and_challenges_project_model():
    prompt = (
        Path(__file__).resolve().parents[1]
        / "mesh"
        / "prompts"
        / "autonomous_controller.txt"
    ).read_text()
    assert "Read the `### Project Model` block inside the Narrative section" in prompt
    assert "descriptive and defeasible, never authority" in prompt
    assert (
        "EVIDENCE=<what proves each selected task done>\n"
        "                note any observations from the Project Model that require "
        "verification"
    ) in prompt


def test_close_prompt_refreshes_project_model_conservatively():
    prompt = (
        Path(__file__).resolve().parents[1]
        / "mesh"
        / "prompts"
        / "autonomous_controller_execute.txt"
    ).read_text()
    assert "**Project Model refresh.**" in prompt
    assert "Add new observations only for verified, material findings." in prompt
    assert "The block must not exceed 1,200 tokens." in prompt
    assert "record the overflow in" in prompt and "leave it unchanged" in prompt
