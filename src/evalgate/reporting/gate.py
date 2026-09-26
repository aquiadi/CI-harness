"""Rendering the gate verdict.

Two outputs from the same result: a markdown table for a pull-request comment
and a JSON file for anything that wants to read the verdict without parsing
prose. The markdown is written to be read in a diff by someone deciding whether
to merge, so it leads with the verdict and the reason, not with a table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalgate.gate.baseline import Baseline
from evalgate.gate.check import Check, Direction, GateResult
from evalgate.gate.items import ItemChange, ItemComparison
from evalgate.hashing import short
from evalgate.reporting.markdown import bullets, heading, number, table
from evalgate.runs.store import RunMeta

TITLE = "Quality gate"
PASS_LINE = "**PASS** -- no threshold exceeded."
FAIL_LINE = "**FAIL** -- this change may not merge as it stands."


def _delta(check: Check) -> str:
    if check.delta is None:
        return "-"
    worse = (check.direction is Direction.HIGHER_IS_BETTER and check.delta < 0) or (
        check.direction is Direction.LOWER_IS_BETTER and check.delta > 0
    )
    return check.delta_label + (" worse" if worse else "")


def _timing(timing: dict[str, Any]) -> str:
    if not timing:
        return "once per question, cold"
    warm = "after a warm-up pass" if timing.get("warmup") else "cold"
    return f"as the {timing.get('statistic', 'median')} of {timing.get('repeats')} repeats, {warm}"


def _number(value: float) -> str:
    return f"{value:g}"


def _items_section(comparison: ItemComparison, listed: int) -> list[str]:
    parts = [heading("Questions that moved", 3)]
    if comparison.unavailable:
        parts.append(f"Not available: {comparison.unavailable}.")
        return parts
    worse, better = comparison.worse, comparison.better
    parts.append(
        f"{len(worse)} score(s) got worse and {len(better)} got better, question by "
        "question. Recall is 1 for a hit and 0 for a miss; judge scores are on the "
        "rubric's scale."
    )
    if worse:
        shown = worse[:listed]
        parts.append(
            table(
                ["question", "measure", "baseline", "this run", "note"],
                [
                    [
                        change.example_id,
                        change.measure,
                        _number(change.before),
                        _number(change.after),
                        "answer text changed" if change.answer_changed else "",
                    ]
                    for change in shown
                ],
            )
        )
        if len(worse) > len(shown):
            parts.append(f"... and {len(worse) - len(shown)} more in reports/gate.json.")
    return parts


def _interval_section(comparison: ItemComparison) -> list[str]:
    interval = comparison.interval
    if interval is None:
        return []
    return [
        heading("How much a sample this size can show", 3),
        f"Composite quality delta {interval.delta:+.4f}; "
        f"{interval.confidence:.0%} paired bootstrap interval "
        f"{interval.low:+.4f} to {interval.high:+.4f}, over {interval.n_retrieval} "
        f"retrieval and {interval.n_answers} answer questions ({interval.resamples} "
        "resamples). This is how far the delta could move under a different sample of "
        "questions of this size. It is reported, not gated: the thresholds above decide "
        "the verdict, and a wide interval is a reason to add questions, not to pass a drop.",
    ]


def render(
    result: GateResult,
    baseline: Baseline,
    meta: RunMeta,
    metrics: dict[str, Any],
    comparison: ItemComparison | None = None,
    items_listed: int = 10,
) -> str:
    """Render the gate result as markdown."""
    parts = [heading(TITLE, 2), FAIL_LINE if not result.passed else PASS_LINE]

    if result.blocking:
        parts.append(heading("Why", 3))
        parts.append(bullets(result.blocking))

    parts.append(
        table(
            ["check", "metric", "compared to", "reference", "this run", "delta", "limit", "status"],
            [
                [
                    check.name,
                    check.metric,
                    "baseline" if check.compared_to == "baseline" else "reference run",
                    number(check.baseline, 6),
                    number(check.current, 6),
                    _delta(check),
                    check.limit_label,
                    check.status,
                ]
                for check in result.checks
            ],
        )
    )
    reference = next(
        (check.compared_to for check in result.checks if check.compared_to != "baseline"), None
    )
    if reference:
        parts.append(
            f"Latency is compared to {reference}: the base commit measured on this runner, "
            "so the check measures the change rather than the difference between machines."
        )
    elif baseline.timing != meta.timing:
        parts.append(
            "Latency is compared to the baseline, which was timed "
            f"{_timing(baseline.timing)} on the machine that froze it; this run was timed "
            f"{_timing(meta.timing)} here. Treat the latency check as a coarse bound. CI "
            "compares against the base commit timed on the same runner instead."
        )

    if comparison is not None:
        parts.extend(_items_section(comparison, items_listed))
        parts.extend(_interval_section(comparison))

    supporting = [
        ("recall@k", "recall_at_k", 3),
        ("nDCG@10", "ndcg_at_10", 3),
        ("MRR", "mrr", 3),
        ("groundedness", "judge_groundedness", 2),
        ("relevance", "judge_relevance", 2),
        ("citation correctness", "judge_citation_correctness", 2),
        ("p50 latency (s)", "p50_latency_s", 4),
        ("context tokens/query", "context_tokens_per_query", 0),
    ]
    rows = []
    for label, key, digits in supporting:
        before = baseline.metrics.get(key)
        after = metrics.get(key)
        if before is None and after is None:
            continue
        rows.append([label, number(before, digits), number(after, digits)])
    if rows:
        parts.append(heading("Supporting metrics (not gated)", 3))
        parts.append(table(["metric", "baseline", "this run"], rows))

    judge = meta.judge or {}
    parts.append(heading("Provenance", 3))
    parts.append(
        bullets(
            [
                f"baseline run: `{baseline.run_id}` frozen {baseline.frozen_at}",
                f"this run: `{meta.run_id}`",
                f"corpus: {meta.corpus_name} ({short(meta.corpus_hash)})",
                f"judge: `{judge.get('model', 'none')}`",
                f"api mode: {meta.api_mode}"
                + (" (replayed from cassettes)" if meta.replayed else ""),
                f"cost compared on `{result.cost_metric}`",
            ]
        )
    )
    return "\n\n".join(parts) + "\n"


# What another run's gate needs from this one to use it as a latency reference,
# and what a reader of the JSON most often wants without opening the run.
RUN_METRICS = (
    "composite_quality",
    "p95_latency_s",
    "p50_latency_s",
    "cost_per_query_usd",
    "projected_cost_per_query_usd",
    "recall_at_k",
)


def _change(change: ItemChange) -> dict[str, Any]:
    return {
        "example_id": change.example_id,
        "measure": change.measure,
        "before": change.before,
        "after": change.after,
        "answer_changed": change.answer_changed,
    }


def write_json(
    path: Path,
    result: GateResult,
    baseline: Baseline,
    meta: RunMeta,
    metrics: dict[str, Any] | None = None,
    comparison: ItemComparison | None = None,
) -> None:
    """Write the verdict in a form other tools can read without parsing prose."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "passed": result.passed,
        "comparable": result.comparable,
        "blocking": result.blocking,
        "cost_metric": result.cost_metric,
        "baseline_run_id": baseline.run_id,
        "run_id": meta.run_id,
        "run_metrics": {key: (metrics or {}).get(key) for key in RUN_METRICS},
        "checks": [
            {
                "name": check.name,
                "metric": check.metric,
                "compared_to": check.compared_to,
                "baseline": check.baseline,
                "current": check.current,
                "delta": check.delta,
                "threshold": check.threshold,
                "absolute": check.absolute,
                "passed": check.passed,
                "reason": check.reason,
            }
            for check in result.checks
        ],
    }
    if comparison is not None:
        interval = comparison.interval
        payload["items"] = {
            "unavailable": comparison.unavailable,
            "worse": [_change(change) for change in comparison.worse],
            "better": [_change(change) for change in comparison.better],
            "interval": None
            if interval is None
            else {
                "delta": interval.delta,
                "low": interval.low,
                "high": interval.high,
                "confidence": interval.confidence,
                "resamples": interval.resamples,
                "n_retrieval": interval.n_retrieval,
                "n_answers": interval.n_answers,
            },
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
