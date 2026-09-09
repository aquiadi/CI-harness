"""Comparing a run to the baseline.

Three checks, all with thresholds from `configs/gate/`: composite quality must
not drop more than a set percentage, p95 latency must not rise more than
another, and cost per query must not rise more than a third.

Two behaviours are deliberate and load-bearing.

*Missing data fails.* A metric absent from the run, a metric absent from the
baseline, or a baseline that does not exist at all is a failure. The state
where a regression is invisible must not be the state where the build is
green.

*Changed ground fails.* If the corpus, the prompts or the eval sets differ from
the baseline's, the comparison is meaningless and the gate says so instead of
producing a number. Changing them is legitimate -- it just requires re-freezing
the baseline deliberately, which is a reviewable diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from evalgate.config import GateConfig
from evalgate.gate.baseline import Baseline
from evalgate.runs.store import RunMeta

QUALITY_METRIC = "composite_quality"
LATENCY_METRIC = "p95_latency_s"
COST_METRIC = "cost_per_query_usd"
PROJECTED_COST_METRIC = "projected_cost_per_query_usd"


class Direction(StrEnum):
    """Which way a metric is allowed to move."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class Check:
    """One threshold comparison."""

    name: str
    metric: str
    direction: Direction
    baseline: float | None
    current: float | None
    delta_pct: float | None
    threshold_pct: float
    passed: bool
    reason: str = ""

    @property
    def status(self) -> str:
        """PASS or FAIL, for tables and logs."""
        return "PASS" if self.passed else "FAIL"


@dataclass(frozen=True, slots=True)
class GateResult:
    """The verdict and everything behind it."""

    checks: list[Check]
    passed: bool
    blocking: list[str]
    cost_metric: str
    comparable: bool

    @property
    def failures(self) -> list[Check]:
        """The checks that failed."""
        return [check for check in self.checks if not check.passed]


def _relative_change(baseline: float, current: float) -> float | None:
    """Percentage change from baseline to current, or None when undefined."""
    if baseline == 0.0:
        return None
    return (current - baseline) / abs(baseline) * 100.0


def _check(
    name: str,
    metric: str,
    direction: Direction,
    baseline_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    threshold_pct: float,
) -> Check:
    baseline_value = baseline_metrics.get(metric)
    current_value = current_metrics.get(metric)
    if baseline_value is None or current_value is None:
        missing = "baseline" if baseline_value is None else "run"
        return Check(
            name=name,
            metric=metric,
            direction=direction,
            baseline=baseline_value,
            current=current_value,
            delta_pct=None,
            threshold_pct=threshold_pct,
            passed=False,
            reason=f"{metric} is missing from the {missing}; missing data fails the gate",
        )

    change = _relative_change(float(baseline_value), float(current_value))
    if change is None:
        # A zero baseline has no percentage change. Any increase away from zero
        # is a regression for a lower-is-better metric; matching zero is fine.
        regressed = direction is Direction.LOWER_IS_BETTER and float(current_value) > 0.0
        return Check(
            name=name,
            metric=metric,
            direction=direction,
            baseline=float(baseline_value),
            current=float(current_value),
            delta_pct=None,
            threshold_pct=threshold_pct,
            passed=not regressed,
            reason=(
                "baseline is zero, so a percentage change is undefined; "
                + ("any increase is treated as a regression" if regressed else "no increase")
            ),
        )

    if direction is Direction.HIGHER_IS_BETTER:
        regressed = -change > threshold_pct
        reason = f"dropped {-change:.2f}% (limit {threshold_pct:.2f}%)" if regressed else ""
    else:
        regressed = change > threshold_pct
        reason = f"rose {change:.2f}% (limit {threshold_pct:.2f}%)" if regressed else ""

    return Check(
        name=name,
        metric=metric,
        direction=direction,
        baseline=float(baseline_value),
        current=float(current_value),
        delta_pct=change,
        threshold_pct=threshold_pct,
        passed=not regressed,
        reason=reason,
    )


def evaluate_gate(
    baseline: Baseline,
    meta: RunMeta,
    metrics: dict[str, Any],
    gate: GateConfig,
) -> GateResult:
    """Compare a run to the baseline and decide whether the build passes."""
    mismatches = baseline.mismatches(meta)
    # Cost is compared on measured spend where the baseline had any, and on the
    # projected figure otherwise -- an offline baseline whose measured cost is
    # zero would otherwise make the cost check vacuous.
    cost_metric = (
        COST_METRIC
        if float(baseline.metrics.get(COST_METRIC, 0.0) or 0.0) > 0.0
        else PROJECTED_COST_METRIC
    )

    checks = [
        _check(
            "quality",
            QUALITY_METRIC,
            Direction.HIGHER_IS_BETTER,
            baseline.metrics,
            metrics,
            gate.quality_drop_pct,
        ),
        _check(
            "p95 latency",
            LATENCY_METRIC,
            Direction.LOWER_IS_BETTER,
            baseline.metrics,
            metrics,
            gate.p95_latency_rise_pct,
        ),
        _check(
            "cost per query",
            cost_metric,
            Direction.LOWER_IS_BETTER,
            baseline.metrics,
            metrics,
            gate.cost_per_query_rise_pct,
        ),
    ]

    blocking = [f"{check.name}: {check.reason}" for check in checks if not check.passed]
    if mismatches:
        blocking.insert(
            0,
            "not comparable to the baseline ("
            + "; ".join(mismatches)
            + "). Re-freeze the baseline deliberately if this change is intended.",
        )

    return GateResult(
        checks=checks,
        passed=not blocking,
        blocking=blocking,
        cost_metric=cost_metric,
        comparable=not mismatches,
    )
