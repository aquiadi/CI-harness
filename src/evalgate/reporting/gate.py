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
from evalgate.gate.check import Direction, GateResult
from evalgate.hashing import short
from evalgate.reporting.markdown import bullets, heading, number, table
from evalgate.runs.store import RunMeta

TITLE = "Quality gate"
PASS_LINE = "**PASS** -- no threshold exceeded."
FAIL_LINE = "**FAIL** -- this change may not merge as it stands."


def _delta(check_delta: float | None, direction: Direction) -> str:
    if check_delta is None:
        return "-"
    arrow = "+" if check_delta > 0 else ""
    marker = ""
    if direction is Direction.HIGHER_IS_BETTER and check_delta < 0:
        marker = " worse"
    if direction is Direction.LOWER_IS_BETTER and check_delta > 0:
        marker = " worse"
    return f"{arrow}{check_delta:.2f}%{marker}"


def render(result: GateResult, baseline: Baseline, meta: RunMeta, metrics: dict[str, Any]) -> str:
    """Render the gate result as markdown."""
    parts = [heading(TITLE, 2), FAIL_LINE if not result.passed else PASS_LINE]

    if result.blocking:
        parts.append(heading("Why", 3))
        parts.append(bullets(result.blocking))

    parts.append(
        table(
            ["check", "metric", "baseline", "this run", "delta", "limit", "status"],
            [
                [
                    check.name,
                    check.metric,
                    number(check.baseline, 6),
                    number(check.current, 6),
                    _delta(check.delta_pct, check.direction),
                    f"{check.threshold_pct:.2f}%",
                    check.status,
                ]
                for check in result.checks
            ],
        )
    )

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

    parts.append(heading("Provenance", 3))
    parts.append(
        bullets(
            [
                f"baseline run: `{baseline.run_id}` frozen {baseline.frozen_at}",
                f"this run: `{meta.run_id}`",
                f"corpus: {meta.corpus_name} ({short(meta.corpus_hash)})",
                f"api mode: {meta.api_mode}"
                + (" (replayed from cassettes)" if meta.replayed else ""),
                f"cost compared on `{result.cost_metric}`",
            ]
        )
    )
    return "\n\n".join(parts) + "\n"


def write_json(path: Path, result: GateResult, baseline: Baseline, meta: RunMeta) -> None:
    """Write the verdict in a form other tools can read without parsing prose."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": result.passed,
        "comparable": result.comparable,
        "blocking": result.blocking,
        "cost_metric": result.cost_metric,
        "baseline_run_id": baseline.run_id,
        "run_id": meta.run_id,
        "checks": [
            {
                "name": check.name,
                "metric": check.metric,
                "baseline": check.baseline,
                "current": check.current,
                "delta_pct": check.delta_pct,
                "threshold_pct": check.threshold_pct,
                "passed": check.passed,
                "reason": check.reason,
            }
            for check in result.checks
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
