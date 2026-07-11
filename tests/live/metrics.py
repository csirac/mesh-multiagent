"""
Metrics collection and trend analysis for live tests.

Provides:
- Cost estimation based on token counts
- Trend comparison against baselines
- Aggregated statistics
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import TestResult, TestRunReport, Grade


# =============================================================================
# Cost Estimation
# =============================================================================

# Approximate costs per 1K tokens (USD) - adjust as needed
TOKEN_COSTS = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-5.1": {"input": 0.01, "output": 0.02},  # Estimate
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "default": {"input": 0.01, "output": 0.02},
}


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "default",
) -> float:
    """Estimate API cost for a given token count."""
    costs = TOKEN_COSTS.get(model, TOKEN_COSTS["default"])
    return (input_tokens / 1000 * costs["input"]) + (output_tokens / 1000 * costs["output"])


def estimate_run_cost(report: TestRunReport, model: str = "default") -> float:
    """Estimate total cost for a test run."""
    # Rough estimate: 70% input, 30% output
    input_tokens = int(report.total_tokens * 0.7)
    output_tokens = int(report.total_tokens * 0.3)
    return estimate_cost(input_tokens, output_tokens, model)


# =============================================================================
# Trend Analysis
# =============================================================================

@dataclass
class TrendMetrics:
    """Comparison between current and baseline metrics."""
    current_value: float
    baseline_value: float
    change_percent: float
    is_regression: bool  # True if change exceeds threshold in wrong direction

    @property
    def change_description(self) -> str:
        """Human-readable change description."""
        direction = "↑" if self.change_percent > 0 else "↓"
        return f"{direction} {abs(self.change_percent):.1f}%"


@dataclass
class TrendReport:
    """Trend analysis comparing current run to baseline."""
    current_run_id: str
    baseline_run_id: str
    duration: TrendMetrics | None = None
    tokens: TrendMetrics | None = None
    llm_calls: TrendMetrics | None = None
    pass_rate: TrendMetrics | None = None
    regressions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON output."""
        def metric_dict(m: TrendMetrics | None) -> dict | None:
            if m is None:
                return None
            return {
                "current": m.current_value,
                "baseline": m.baseline_value,
                "change_percent": m.change_percent,
                "is_regression": m.is_regression,
            }

        return {
            "current_run_id": self.current_run_id,
            "baseline_run_id": self.baseline_run_id,
            "duration": metric_dict(self.duration),
            "tokens": metric_dict(self.tokens),
            "llm_calls": metric_dict(self.llm_calls),
            "pass_rate": metric_dict(self.pass_rate),
            "regressions": self.regressions,
        }


def compute_trend(
    current: float,
    baseline: float,
    regression_threshold: float = 0.2,  # 20% change = regression
    higher_is_better: bool = False,
) -> TrendMetrics:
    """Compute trend metrics for a single value."""
    if baseline == 0:
        change_percent = 0.0 if current == 0 else 100.0
    else:
        change_percent = ((current - baseline) / baseline) * 100

    # Determine if this is a regression
    if higher_is_better:
        is_regression = change_percent < -regression_threshold * 100
    else:
        is_regression = change_percent > regression_threshold * 100

    return TrendMetrics(
        current_value=current,
        baseline_value=baseline,
        change_percent=change_percent,
        is_regression=is_regression,
    )


def compare_runs(
    current: TestRunReport,
    baseline: TestRunReport,
    regression_threshold: float = 0.2,
) -> TrendReport:
    """Compare current run to baseline and identify regressions."""
    report = TrendReport(
        current_run_id=current.run_id,
        baseline_run_id=baseline.run_id,
    )

    # Compare duration (lower is better)
    report.duration = compute_trend(
        current.total_duration_seconds,
        baseline.total_duration_seconds,
        regression_threshold,
        higher_is_better=False,
    )

    # Compare tokens (lower is better)
    report.tokens = compute_trend(
        current.total_tokens,
        baseline.total_tokens,
        regression_threshold,
        higher_is_better=False,
    )

    # Compare LLM calls (lower is better)
    report.llm_calls = compute_trend(
        current.total_llm_calls,
        baseline.total_llm_calls,
        regression_threshold,
        higher_is_better=False,
    )

    # Compare pass rate (higher is better)
    current_rate = current.passed / current.total_tests if current.total_tests > 0 else 0
    baseline_rate = baseline.passed / baseline.total_tests if baseline.total_tests > 0 else 0
    report.pass_rate = compute_trend(
        current_rate,
        baseline_rate,
        regression_threshold,
        higher_is_better=True,
    )

    # Collect regressions
    if report.duration and report.duration.is_regression:
        report.regressions.append(f"Duration: {report.duration.change_description}")
    if report.tokens and report.tokens.is_regression:
        report.regressions.append(f"Tokens: {report.tokens.change_description}")
    if report.llm_calls and report.llm_calls.is_regression:
        report.regressions.append(f"LLM calls: {report.llm_calls.change_description}")
    if report.pass_rate and report.pass_rate.is_regression:
        report.regressions.append(f"Pass rate: {report.pass_rate.change_description}")

    return report


# =============================================================================
# Baseline Management
# =============================================================================

def load_baseline(baselines_dir: Path) -> TestRunReport | None:
    """Load the most recent baseline report."""
    if not baselines_dir.exists():
        return None

    # Find most recent baseline
    baselines = sorted(baselines_dir.glob("*.json"), reverse=True)
    if not baselines:
        return None

    with open(baselines[0]) as f:
        data = json.load(f)

    # Reconstruct report from JSON
    report = TestRunReport(
        run_id=data["run_id"],
        timestamp=data["timestamp"],
    )
    report.total_tests = data["summary"]["total"]
    report.passed = data["summary"]["passed"]
    report.partial = data["summary"]["partial"]
    report.failed = data["summary"]["failed"]
    report.total_duration_seconds = data["metrics"]["total_duration_seconds"]
    report.total_tokens = data["metrics"]["total_tokens"]
    report.total_llm_calls = data["metrics"]["total_llm_calls"]

    return report


def save_as_baseline(report: TestRunReport, baselines_dir: Path) -> Path:
    """Save a report as a new baseline."""
    baselines_dir.mkdir(parents=True, exist_ok=True)
    filename = f"baseline-{report.run_id}.json"
    path = baselines_dir / filename

    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    return path


# =============================================================================
# Aggregate Statistics
# =============================================================================

@dataclass
class CategoryStats:
    """Aggregate stats for a test category."""
    category: str
    total: int = 0
    passed: int = 0
    partial: int = 0
    failed: int = 0
    avg_duration: float = 0.0
    avg_tokens: int = 0
    avg_llm_calls: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


def compute_category_stats(results: list[TestResult]) -> dict[str, CategoryStats]:
    """Compute stats grouped by category."""
    by_category: dict[str, list[TestResult]] = {}
    for r in results:
        by_category.setdefault(r.category, []).append(r)

    stats: dict[str, CategoryStats] = {}
    for category, cat_results in by_category.items():
        s = CategoryStats(category=category)
        s.total = len(cat_results)
        s.passed = sum(1 for r in cat_results if r.grade == Grade.PASS)
        s.partial = sum(1 for r in cat_results if r.grade == Grade.PARTIAL)
        s.failed = sum(1 for r in cat_results if r.grade == Grade.FAIL)
        s.avg_duration = sum(r.total_duration_seconds for r in cat_results) / s.total
        s.avg_tokens = int(sum(r.total_tokens for r in cat_results) / s.total)
        s.avg_llm_calls = sum(r.llm_calls for r in cat_results) / s.total
        stats[category] = s

    return stats


def print_category_stats(stats: dict[str, CategoryStats]) -> None:
    """Print category statistics table."""
    print("\nCategory Statistics:")
    print("-" * 80)
    print(f"{'Category':<12} {'Total':<6} {'Pass':<6} {'Partial':<8} {'Fail':<6} {'Rate':<8} {'Avg Time':<10}")
    print("-" * 80)

    for cat in ["trivial", "simple", "moderate", "complex", "edge"]:
        if cat not in stats:
            continue
        s = stats[cat]
        print(f"{cat:<12} {s.total:<6} {s.passed:<6} {s.partial:<8} {s.failed:<6} "
              f"{s.pass_rate:.1%:<8} {s.avg_duration:.1f}s")

    print("-" * 80)
