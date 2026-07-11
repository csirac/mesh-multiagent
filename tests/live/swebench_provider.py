"""
SWE-bench Pro provider for mesh agent evaluation.

Loads problems from HuggingFace (ScaleAI/SWE-bench_Pro), formats task
prompts for mesh agents, and collects git diff patches as solutions.

The agent receives:
- problem_statement (the GitHub issue)
- interface (expected function/class signatures)
- requirements (behavioral expectations)
- repo info (repo name, language, base commit)

The agent produces: a git diff/patch that resolves the issue.

Evaluation requires Docker (via SWE-bench eval harness) and is handled
separately from patch generation. This module focuses on the agent-side
workflow.

Usage:
    from tests.live.swebench_provider import (
        load_problems, build_agentic_prompt, extract_patch,
    )

    problems = load_problems(max_problems=5, language="python")
    prompt = build_agentic_prompt(problems[0])
    patch = extract_patch(agent_response)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SWEBenchProblem:
    """A SWE-bench Pro task instance."""
    instance_id: str
    repo: str
    repo_language: str
    base_commit: str
    problem_statement: str
    interface: str
    requirements: str
    patch: str  # gold patch (for reference, not shown to agent)
    test_patch: str  # test changes (not shown to agent)
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    issue_specificity: list[str]
    issue_categories: list[str]
    before_repo_set_cmd: str
    selected_test_files_to_run: list[str]

    @property
    def repo_owner(self) -> str:
        return self.repo.split("/")[0]

    @property
    def repo_name(self) -> str:
        return self.repo.split("/")[1]

    @property
    def github_url(self) -> str:
        return f"https://github.com/{self.repo}"

    @property
    def patch_file_count(self) -> int:
        """Number of files changed in the gold patch."""
        return len(re.findall(r'^diff --git', self.patch, re.MULTILINE))

    @property
    def patch_line_count(self) -> int:
        """Lines in the gold patch."""
        return self.patch.count('\n')

    @property
    def difficulty_estimate(self) -> str:
        """Rough difficulty based on patch size."""
        files = self.patch_file_count
        lines = self.patch_line_count
        if files <= 1 and lines <= 30:
            return "easy"
        elif files <= 3 and lines <= 100:
            return "medium"
        else:
            return "hard"


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

def load_problems(
    max_problems: int | None = None,
    language: str | None = None,
    repo: str | None = None,
    difficulty: str | None = None,
    sort_by: str = "patch_size",
) -> list[SWEBenchProblem]:
    """
    Load SWE-bench Pro problems from HuggingFace.

    Args:
        max_problems: Limit number of problems
        language: Filter by language ("python", "js", "go", "ts")
        repo: Filter by repo name (e.g., "ansible/ansible")
        difficulty: Filter by estimated difficulty ("easy", "medium", "hard")
        sort_by: How to sort problems before limiting:
            - "patch_size": Smallest patches first (easier problems first)
            - "instance_id": Alphabetical
            - "none": Dataset order

    Returns:
        List of SWEBenchProblem objects
    """
    from datasets import load_dataset

    logger.info("Loading SWE-bench Pro dataset from HuggingFace...")
    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    logger.info(f"Loaded {len(ds)} total instances")

    problems = []
    for row in ds:
        # Parse list fields that may be JSON strings
        fail_to_pass = _parse_list_field(row.get("fail_to_pass", []))
        pass_to_pass = _parse_list_field(row.get("pass_to_pass", []))
        issue_specificity = _parse_list_field(row.get("issue_specificity", []))
        issue_categories = _parse_list_field(row.get("issue_categories", []))
        selected_test_files = _parse_list_field(row.get("selected_test_files_to_run", []))

        prob = SWEBenchProblem(
            instance_id=row["instance_id"],
            repo=row["repo"],
            repo_language=row["repo_language"],
            base_commit=row["base_commit"],
            problem_statement=row["problem_statement"],
            interface=row.get("interface", ""),
            requirements=row.get("requirements", ""),
            patch=row["patch"],
            test_patch=row.get("test_patch", ""),
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            issue_specificity=issue_specificity,
            issue_categories=issue_categories,
            before_repo_set_cmd=row.get("before_repo_set_cmd", ""),
            selected_test_files_to_run=selected_test_files,
        )
        problems.append(prob)

    # Apply filters
    if language:
        problems = [p for p in problems if p.repo_language == language.lower()]
        logger.info(f"Filtered to {len(problems)} {language} problems")

    if repo:
        problems = [p for p in problems if p.repo == repo]
        logger.info(f"Filtered to {len(problems)} problems from {repo}")

    if difficulty:
        problems = [p for p in problems if p.difficulty_estimate == difficulty.lower()]
        logger.info(f"Filtered to {len(problems)} {difficulty} problems")

    # Sort
    if sort_by == "patch_size":
        problems.sort(key=lambda p: len(p.patch))
    elif sort_by == "instance_id":
        problems.sort(key=lambda p: p.instance_id)

    if max_problems:
        problems = problems[:max_problems]

    logger.info(f"Returning {len(problems)} problems")
    return problems


def _parse_list_field(val) -> list[str]:
    """Parse a field that might be a JSON string or already a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return [val] if val else []
    return []


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def build_agentic_prompt(problem: SWEBenchProblem, include_repo_clone: bool = True) -> str:
    """
    Build an agentic prompt for a SWE-bench task.

    The agent will receive the issue, interface, and requirements, then use
    tools to clone the repo, explore code, and generate a fix.

    Args:
        problem: The SWE-bench problem
        include_repo_clone: Whether to include repo clone instructions

    Returns:
        Formatted prompt string
    """
    prompt = "IGNORE ALL PREVIOUS CONTEXT. This is a new, independent task.\n\n"
    prompt += "# SWE-bench Task: Fix a GitHub Issue\n\n"
    prompt += (
        "You are an expert software engineer. Your task is to fix the following "
        "GitHub issue by producing a git diff patch.\n\n"
    )

    prompt += f"## Repository\n"
    prompt += f"- **Repo**: {problem.repo}\n"
    prompt += f"- **Language**: {problem.repo_language}\n"
    prompt += f"- **Base commit**: `{problem.base_commit}`\n"
    prompt += f"- **GitHub URL**: {problem.github_url}\n\n"

    prompt += f"## Issue Description\n\n{problem.problem_statement}\n\n"

    if problem.interface:
        prompt += f"## Interface\n\n{problem.interface}\n\n"

    if problem.requirements:
        prompt += f"## Requirements\n\n{problem.requirements}\n\n"

    # Failing tests (the agent should know what tests need to pass)
    if problem.fail_to_pass:
        prompt += "## Tests That Must Pass After Your Fix\n\n"
        for test in problem.fail_to_pass[:10]:  # Cap at 10 to avoid prompt bloat
            prompt += f"- `{test}`\n"
        if len(problem.fail_to_pass) > 10:
            prompt += f"- ... and {len(problem.fail_to_pass) - 10} more\n"
        prompt += "\n"

    if problem.selected_test_files_to_run:
        prompt += "## Test Files to Run\n\n"
        for tf in problem.selected_test_files_to_run:
            prompt += f"- `{tf}`\n"
        prompt += "\n"

    prompt += "## Instructions\n\n"

    if include_repo_clone:
        # Use a unique working directory per task to avoid stale repo issues
        workdir = f"/tmp/swe_{problem.repo_name}_{problem.base_commit[:8]}"
        prompt += (
            "1. **IMPORTANT: Clean workspace and clone the repo**:\n"
            "   ```bash\n"
            f"   rm -rf {workdir}\n"
            f"   git clone {problem.github_url} {workdir} 2>&1\n"
            f"   cd {workdir} && git checkout {problem.base_commit}\n"
            "   ```\n\n"
        )
        prompt += (
            "2. **Explore the codebase** to understand the issue. Read the relevant files "
            "mentioned in the interface and requirements.\n\n"
            "3. **Implement the fix** by editing the necessary files.\n\n"
            "4. **Generate the patch** with:\n"
            "   ```bash\n"
            f"   cd {workdir} && git diff\n"
            "   ```\n\n"
        )
    else:
        prompt += (
            "1. **Analyze the issue** based on the problem statement, interface, and requirements.\n\n"
            "2. **Reason about the fix** — what files need to change and how.\n\n"
            "3. **Produce a git diff** patch that resolves the issue.\n\n"
        )

    prompt += (
        "## CRITICAL: Output Requirement\n\n"
        "After implementing your fix, you MUST:\n"
        "1. Run `cd /tmp/swebench_repo && git diff` to generate the patch\n"
        "2. Include the COMPLETE output in your final message inside a ```diff code block\n\n"
        "Do NOT just say 'Done' or describe changes in prose. "
        "You MUST paste the full `git diff` output.\n\n"
        "Example format:\n"
        "```diff\n"
        "diff --git a/path/to/file.py b/path/to/file.py\n"
        "--- a/path/to/file.py\n"
        "+++ b/path/to/file.py\n"
        "@@ -10,6 +10,7 @@\n"
        " context line\n"
        "-removed line\n"
        "+added line\n"
        "```\n\n"
        "The patch should be minimal — only change what is necessary to fix the issue.\n"
        "Do NOT include test file changes in your patch (tests are provided separately).\n"
    )

    return prompt


def build_simple_prompt(problem: SWEBenchProblem) -> str:
    """
    Build a non-agentic prompt (for direct/passthrough mode).

    The agent generates a patch based only on the issue description,
    without cloning the repo or using tools.
    """
    prompt = build_agentic_prompt(problem, include_repo_clone=False)
    return prompt


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patch(response: str) -> str:
    """
    Extract a git diff patch from an agent response.

    Looks for fenced code blocks with diff content and combines them
    into a single patch.

    Returns empty string if no valid patch found.
    """
    if not response:
        return ""

    # Strategy 1: Fenced ```diff blocks
    diff_blocks = re.findall(r'```(?:diff|patch)?\n?(.*?)```', response, re.DOTALL)
    patches = []
    for block in diff_blocks:
        block = block.strip()
        if _looks_like_diff(block):
            patches.append(block)

    if patches:
        return "\n".join(patches)

    # Strategy 2: Look for unfenced diff content
    # Find lines starting with "diff --git"
    lines = response.split("\n")
    in_diff = False
    diff_lines = []
    for line in lines:
        if line.startswith("diff --git"):
            in_diff = True
            diff_lines.append(line)
        elif in_diff:
            if line.startswith(("---", "+++", "@@", " ", "+", "-", "\\")):
                diff_lines.append(line)
            elif line.startswith("diff --git"):
                diff_lines.append(line)
            else:
                # End of diff section
                if not line.strip():
                    diff_lines.append(line)
                else:
                    in_diff = False

    if diff_lines:
        result = "\n".join(diff_lines).strip()
        if _looks_like_diff(result):
            return result

    return ""


def _looks_like_diff(text: str) -> bool:
    """Check if text looks like a valid git diff."""
    if not text:
        return False
    # Must contain at least one diff header or hunk header
    has_diff_header = "diff --git" in text or "---" in text
    has_changes = ("+" in text or "-" in text)
    return has_diff_header and has_changes


def validate_patch(patch: str, problem: SWEBenchProblem) -> dict:
    """
    Basic validation of a generated patch.

    Returns dict with:
        - valid: bool
        - files_changed: list of file paths
        - lines_added: int
        - lines_removed: int
        - issues: list of warning strings
    """
    result = {
        "valid": False,
        "files_changed": [],
        "lines_added": 0,
        "lines_removed": 0,
        "issues": [],
    }

    if not patch.strip():
        result["issues"].append("Empty patch")
        return result

    # Parse file paths
    files = re.findall(r'^diff --git a/(.+?) b/', patch, re.MULTILINE)
    result["files_changed"] = files

    # Count changes
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            result["lines_added"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            result["lines_removed"] += 1

    # Validation checks
    if not files:
        result["issues"].append("No files changed in patch")
        return result

    # Check for test file contamination
    test_files = [f for f in files if "test" in f.lower()]
    if test_files:
        result["issues"].append(f"Patch modifies test files: {test_files}")

    # Basic format check
    if "@@" not in patch:
        result["issues"].append("No hunk headers (@@ ... @@) found")

    result["valid"] = len(result["issues"]) == 0 or (
        len(result["issues"]) == 1 and "test files" in result["issues"][0]
    )

    return result


# ---------------------------------------------------------------------------
# SWE-bench prediction format
# ---------------------------------------------------------------------------

def format_prediction(instance_id: str, model_patch: str, model_name: str = "mesh-agent") -> dict:
    """
    Format a prediction in SWE-bench submission format.

    This is the format expected by the SWE-bench evaluation harness.
    """
    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "model_name_or_path": model_name,
    }
