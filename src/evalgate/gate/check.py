"""Comparing a run to the baseline.

Three headline checks, all with thresholds from `configs/gate/`: composite
quality must not drop more than a set percentage, p95 latency must not rise
more than another, and cost per query must not rise more than a third. Under
them, every component of the composite has a floor of its own, because a
composite can hold steady while one of its parts collapses and another rises
to cover it -- citation correctness falling while recall improves reads as no
change at all.

Two behaviours are deliberate and load-bearing.

*Missing data fails.* A metric absent from the run, a metric absent from the
baseline, or a baseline that does not exist at all is a failure. The state
where a regression is invisible must not be the state where the build is
green.

*Changed ground fails.* If the corpus, the prompts, the eval sets or the judge
differ from the baseline's, the comparison is meaningless and the gate says so
instead of producing a number. Changing them is legitimate -- it just requires
re-freezing the baseline deliberately, which is a reviewable diff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from evalgate.config import GateConfig
from evalgate.gate.baseline import Baseline
from evalgate.runs.store import RunMeta

QUALITY_METRIC = "composite_quality"
LATENCY_METRIC = "p95_latency_s"
COST_METRIC = "cost_per_query_usd"
PROJECTED_COST_METRIC = "projected_cost_per_query_usd"
COMPONENTS_METRIC = "composite_components"
COMPONENT_PREFIX = "component: "


class Direction(StrEnum):
    """Which way a metric is allowed to move."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class Check:
    """One threshold comparison.

    `delta` and `threshold` are percentages of the baseline when `absolute` is
    false, and plain differences on the metric's own scale when it is true --
    component scores are already normalised to 0..1, where a drop of 0.1 means
    the same thing at any starting point and a percentage does not.
    """

    name: str
    metric: str
    direction: Direction
    baseline: float | None
    current: float | None
    delta: float | None
    threshold: float
    passed: bool
    reason: str = ""
    absolute: bool = False
    compared_to: str = "baseline"

    @property
    def status(self) -> str:
        """PASS or FAIL, for tables and logs."""
        return "PASS" if self.passed else "FAIL"

    @property
    def delta_label(self) -> str:
        """The change, in the check's own unit."""
        if self.delta is None:
            return "-"
        return f"{self.delta:+.3f}" if self.absolute else f"{self.delta:+.2f}%"

    @property
    def limit_label(self) -> str:
        """The threshold, in the check's own unit."""
        return f"{self.threshold:.3f}" if self.absolute else f"{self.threshold:.2f}%"


@dataclass(frozen=True, slots=True)
class LatencyReference:
    """p95 latency measured somewhere other than the baseline.

    The baseline's latency was measured on whatever machine froze it. CI runs
    on another, so comparing the two measures the hardware as much as the
    change. When CI measures the base commit on the same runner first, that is
    the number a pull request's latency should be held to.
    """

    source: str
    p95_latency_s: float | None


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


def load_latency_reference(path: Path) -> LatencyReference:
    """Read a reference p95 from another run's gate.json or metrics.json.

    Accepts the gate.json this module writes (`run_metrics`), the older shape
    that only recorded the latency check's current value, and a run's
    metrics.json -- because the reference comes from the base commit, whose
    code may predate the current format. An unreadable reference is returned
    with no value, and the latency check then fails: missing data fails.
    """
    source = str(path)
    if not path.is_file():
        return LatencyReference(source=source, p95_latency_s=None)
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return LatencyReference(source=source, p95_latency_s=None)
    candidates: list[Any] = [
        payload.get("run_metrics", {}).get(LATENCY_METRIC),
        payload.get(LATENCY_METRIC),
        *[
            check.get("current")
            for check in payload.get("checks", [])
            if isinstance(check, dict) and check.get("metric") == LATENCY_METRIC
        ],
    ]
    value = next((item for item in candidates if isinstance(item, int | float)), None)
    return LatencyReference(source=source, p95_latency_s=None if value is None else float(value))


def _relative_change(baseline: float, current: float) -> float | None:
    """Percentage change from baseline to current, or None when undefined."""
    if baseline == 0.0:
        return None
    return (current - baseline) / abs(baseline) * 100.0


def _missing(
    name: str,
    metric: str,
    direction: Direction,
    baseline_value: Any,
    current_value: Any,
    threshold: float,
    absolute: bool,
    compared_to: str,
) -> Check:
    missing = compared_to if baseline_value is None else "run"
    return Check(
        name=name,
        metric=metric,
        direction=direction,
        baseline=baseline_value,
        current=current_value,
        delta=None,
        threshold=threshold,
        passed=False,
        reason=f"{metric} is missing from the {missing}; missing data fails the gate",
        absolute=absolute,
        compared_to=compared_to,
    )


def _check(
    name: str,
    metric: str,
    direction: Direction,
    baseline_value: Any,
    current_value: Any,
    threshold_pct: float,
    compared_to: str = "baseline",
) -> Check:
    if baseline_value is None or current_value is None:
        return _missing(
            name,
            metric,
            direction,
            baseline_value,
            current_value,
            threshold_pct,
            False,
            compared_to,
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
            delta=None,
            threshold=threshold_pct,
            passed=not regressed,
            reason=(
                "baseline is zero, so a percentage change is undefined; "
                + ("any increase is treated as a regression" if regressed else "no increase")
            ),
            compared_to=compared_to,
        )

    if direction is Direction.HIGHER_IS_BETTER:
        regressed = -change > threshold_pct
        reason = f"dropped {-change:.2f}% (limit {threshold_pct:.2f}%)" if regressed else ""
    else:
        regressed = change > threshold_pct
        reason = f"rose {change:.2f}% (limit {threshold_pct:.2f}%)" if regressed else ""
    if regressed and compared_to != "baseline":
        reason += f" against {compared_to}"

    return Check(
        name=name,
        metric=metric,
        direction=direction,
        baseline=float(baseline_value),
        current=float(current_value),
        delta=change,
        threshold=threshold_pct,
        passed=not regressed,
        reason=reason,
        compared_to=compared_to,
    )


def _component_check(
    component: str, baseline_value: Any, current_value: Any, max_drop: float
) -> Check:
    """One composite component may not fall by more than `max_drop` points."""
    name = f"{COMPONENT_PREFIX}{component}"
    metric = f"{COMPONENTS_METRIC}.{component}"
    if baseline_value is None or current_value is None:
        return _missing(
            name,
            metric,
            Direction.HIGHER_IS_BETTER,
            baseline_value,
            current_value,
            max_drop,
            True,
            "baseline",
        )
    delta = float(current_value) - float(baseline_value)
    regressed = -delta > max_drop
    return Check(
        name=name,
        metric=metric,
        direction=Direction.HIGHER_IS_BETTER,
        baseline=float(baseline_value),
        current=float(current_value),
        delta=delta,
        threshold=max_drop,
        passed=not regressed,
        reason=f"dropped {-delta:.3f} (limit {max_drop:.3f})" if regressed else "",
        absolute=True,
    )


def _components(metrics: dict[str, Any]) -> dict[str, Any]:
    node = metrics.get(COMPONENTS_METRIC)
    return node if isinstance(node, dict) else {}


def evaluate_gate(
    baseline: Baseline,
    meta: RunMeta,
    metrics: dict[str, Any],
    gate: GateConfig,
    latency_reference: LatencyReference | None = None,
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

    if latency_reference is None:
        latency = _check(
            "p95 latency",
            LATENCY_METRIC,
            Direction.LOWER_IS_BETTER,
            baseline.metrics.get(LATENCY_METRIC),
            metrics.get(LATENCY_METRIC),
            gate.p95_latency_rise_pct,
        )
    else:
        latency = _check(
            "p95 latency",
            LATENCY_METRIC,
            Direction.LOWER_IS_BETTER,
            latency_reference.p95_latency_s,
            metrics.get(LATENCY_METRIC),
            gate.p95_latency_rise_pct,
            compared_to=f"the reference run ({latency_reference.source})",
        )

    checks = [
        _check(
            "quality",
            QUALITY_METRIC,
            Direction.HIGHER_IS_BETTER,
            baseline.metrics.get(QUALITY_METRIC),
            metrics.get(QUALITY_METRIC),
            gate.quality_drop_pct,
        ),
        latency,
        _check(
            "cost per query",
            cost_metric,
            Direction.LOWER_IS_BETTER,
            baseline.metrics.get(cost_metric),
            metrics.get(cost_metric),
            gate.cost_per_query_rise_pct,
        ),
    ]
    before = _components(baseline.metrics)
    after = _components(metrics)
    checks.extend(
        _component_check(
            component, before.get(component), after.get(component), gate.component_drop_max
        )
        for component in sorted(gate.composite_weights)
    )

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
