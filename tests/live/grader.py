"""
Test grading logic for the live testing framework.

Implements structural grading based on deterministic criteria:
- Required phases present
- Phase budget not exceeded
- Required tools called
- Forbidden tools not called
- Response keyword checks
- Resource limits (duration, tokens, LLM calls)
"""

from .models import Grade, TestExpectation, TestResult


def grade_test(result: TestResult, expected: TestExpectation) -> tuple[Grade, list[str]]:
    """
    Grade a test result against expectations.

    Returns:
        (Grade, list of issues)

    Grading logic:
        PASS:    All constraints met
        PARTIAL: Completed but exceeded budgets or has warnings
        FAIL:    Missing required phases/tools, errors when not expected
    """
    issues: list[str] = []
    has_failure = False
    has_warning = False

    # Check for unexpected errors
    if expected.no_errors and result.has_errors:
        issues.append(f"Unexpected error: {result.error_message}")
        has_failure = True

    # Check required phases
    for phase in expected.min_phases:
        if phase not in result.phases_triggered:
            issues.append(f"Missing required phase: {phase}")
            has_failure = True

    # Check phase budget
    if len(result.phases_triggered) > expected.max_phases:
        issues.append(
            f"Phase budget exceeded: {len(result.phases_triggered)} > {expected.max_phases}"
        )
        has_warning = True

    # Check required tools
    tools_called = result.get_tool_names()
    for tool in expected.tools_required:
        if tool not in tools_called:
            issues.append(f"Missing required tool: {tool}")
            has_failure = True

    # Check forbidden tools
    for tool in expected.tools_forbidden:
        if tool in tools_called:
            issues.append(f"Forbidden tool called: {tool}")
            has_failure = True

    # Check response keywords (contains)
    response_lower = result.response_text.lower()
    for keyword in expected.response_contains:
        if keyword.lower() not in response_lower:
            issues.append(f"Response missing keyword: '{keyword}'")
            has_warning = True

    # Check response keywords (excludes)
    for keyword in expected.response_excludes:
        if keyword.lower() in response_lower:
            issues.append(f"Response contains forbidden keyword: '{keyword}'")
            has_warning = True

    # Check duration budget
    if result.total_duration_seconds > expected.max_duration_seconds:
        issues.append(
            f"Duration exceeded: {result.total_duration_seconds:.1f}s > {expected.max_duration_seconds}s"
        )
        has_warning = True

    # Check token budget
    if result.total_tokens > expected.max_tokens:
        issues.append(
            f"Token budget exceeded: {result.total_tokens} > {expected.max_tokens}"
        )
        has_warning = True

    # Check LLM call budget
    if result.llm_calls > expected.max_llm_calls:
        issues.append(
            f"LLM call budget exceeded: {result.llm_calls} > {expected.max_llm_calls}"
        )
        has_warning = True

    # Determine grade
    if has_failure:
        grade = Grade.FAIL
    elif has_warning:
        grade = Grade.PARTIAL
    else:
        grade = Grade.PASS

    return grade, issues


def format_grade_report(result: TestResult) -> str:
    """Format a human-readable grade report for a single test."""
    lines = [
        f"[{result.grade.value}] {result.scenario_id} ({result.category})",
        f"  Duration: {result.total_duration_seconds:.1f}s | Tokens: {result.total_tokens} | LLM calls: {result.llm_calls}",
        f"  Phases: {' → '.join(result.phases_triggered) if result.phases_triggered else 'none'}",
        f"  Tools: {', '.join(result.get_tool_names()) if result.tools_called else 'none'}",
    ]

    if result.grade_issues:
        lines.append("  Issues:")
        for issue in result.grade_issues:
            lines.append(f"    - {issue}")

    return "\n".join(lines)
