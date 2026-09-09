"""The ablation report and the Pareto frontier.

Reads every run directory, refuses to put incomparable runs in one table, and
writes the full results plus two frontier figures: quality against cost and
quality against p95 latency.

The cost column needs care. `cost_usd` is measured from API usage and is
therefore zero for a generator that makes no API call. `projected_usd` prices
the same run's context and answer tokens at the configured model's rates -- what
this configuration would cost per query if it ran through the API. They are
different quantities and the table keeps them in different columns; the cost
frontier is plotted on whichever one actually varies, and the caption says
which.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalgate.hashing import short
from evalgate.reporting.markdown import bullets, heading, number, table
from evalgate.reporting.plots import Point, frontier, render
from evalgate.runs.store import RunMeta, list_runs, load_meta, load_metrics

TITLE = "Retrieval ablations and the cost/latency/quality frontier"
FIGURES_DIR = "figures"
COST_FIGURE = "pareto_quality_cost.png"
LATENCY_FIGURE = "pareto_quality_latency.png"
MS = 1000.0


@dataclass(frozen=True, slots=True)
class RunSummary:
    """One measured configuration."""

    run_id: str
    meta: RunMeta
    metrics: dict[str, Any]

    @property
    def label(self) -> str:
        """Compact configuration label used in tables and on the plots."""
        fingerprint = self.meta.fingerprint
        chunker = str(fingerprint.get("chunker", {}).get("name", "?"))
        retriever = str(fingerprint.get("retriever", {}).get("name", "?"))
        return f"{_abbreviate(chunker)}/{retriever}/k={self.metrics.get('k', '?')}"

    @property
    def quality(self) -> float:
        """Composite quality as defined by the gate weights."""
        return float(self.metrics["composite_quality"])

    @property
    def p95_ms(self) -> float:
        """p95 serving latency in milliseconds."""
        return float(self.metrics["p95_latency_s"]) * MS

    @property
    def measured_cost(self) -> float:
        """Measured USD per query, from API usage."""
        return float(self.metrics.get("cost_per_query_usd", 0.0))

    @property
    def projected_cost(self) -> float:
        """USD per query this configuration would cost through the API."""
        return float(self.metrics.get("projected_cost_per_query_usd", 0.0))


def _abbreviate(chunker: str) -> str:
    return {
        "recursive_structural": "recursive",
        "section_aware": "section",
        "fixed_token": "fixed",
    }.get(chunker, chunker)


def load_summaries(runs_root: Path) -> list[RunSummary]:
    """Load every run under the runs directory."""
    summaries: list[RunSummary] = []
    for path in list_runs(runs_root):
        meta = load_meta(path)
        summaries.append(RunSummary(run_id=path.name, meta=meta, metrics=load_metrics(path)))
    return summaries


def latest_per_config(summaries: Sequence[RunSummary]) -> list[RunSummary]:
    """One row per configuration: the most recent measurement of each.

    Re-measuring a configuration is normal -- freezing a baseline does it, and
    so does re-running a cell after a change. The runs are all kept on disk,
    but the table shows the current measurement of each configuration rather
    than the same cell twice.
    """
    newest: dict[str, RunSummary] = {}
    for summary in sorted(summaries, key=lambda item: item.run_id):
        newest[summary.meta.config_hash] = summary
    return sorted(newest.values(), key=lambda item: item.run_id)


def partition_comparable(
    summaries: Sequence[RunSummary],
) -> tuple[list[RunSummary], list[RunSummary]]:
    """Split runs into the largest comparable group and everything else.

    Runs are comparable only when corpus, prompts, index inputs and eval sets
    match. Rather than footnote a mixed table, the report tabulates the largest
    comparable group and lists the rest as excluded, with what differs.
    """
    groups: dict[tuple[str, str, str], list[RunSummary]] = {}
    for summary in summaries:
        key = (
            summary.meta.corpus_hash,
            _hash_of(summary.meta.prompt_hashes),
            _hash_of(summary.meta.evalset_hashes),
        )
        groups.setdefault(key, []).append(summary)
    if not groups:
        return [], []
    largest = max(groups.values(), key=len)
    excluded = [summary for summary in summaries if summary not in largest]
    return largest, excluded


def _hash_of(mapping: dict[str, str]) -> str:
    from evalgate.hashing import hash_obj

    return hash_obj(mapping)


def _results_table(summaries: Sequence[RunSummary], frontier_labels: set[str]) -> str:
    headers = [
        "",
        "config",
        "quality",
        "recall@k",
        "nDCG@10",
        "MRR",
        "grounded",
        "relevant",
        "citations",
        "p50 ms",
        "p95 ms",
        "ctx tok",
        "measured $/q",
        "projected $/q",
    ]
    rows = []
    for summary in sorted(summaries, key=lambda item: -item.quality):
        metrics = summary.metrics
        rows.append(
            [
                "*" if summary.label in frontier_labels else "",
                summary.label,
                number(summary.quality),
                number(metrics.get("recall_at_k")),
                number(metrics.get("ndcg_at_10")),
                number(metrics.get("mrr")),
                number(metrics.get("judge_groundedness"), 2),
                number(metrics.get("judge_relevance"), 2),
                number(metrics.get("judge_citation_correctness"), 2),
                number(float(metrics.get("p50_latency_s", 0.0)) * MS, 1),
                number(summary.p95_ms, 1),
                number(metrics.get("context_tokens_per_query"), 0),
                number(summary.measured_cost, 6),
                number(summary.projected_cost, 6),
            ]
        )
    return table(headers, rows)


def build_report(
    summaries: Sequence[RunSummary],
    excluded: Sequence[RunSummary],
    reports_dir: Path,
) -> str:
    """Render the ablation report and write its figures."""
    parts: list[str] = [heading(TITLE, 1)]

    if not summaries:
        parts.append("No runs under `runs/`. Run `make ablate`.")
        return "\n\n".join(parts) + "\n"

    reference = summaries[0].meta
    parts.append(
        bullets(
            [
                f"configurations measured: {len(summaries)}",
                f"corpus: {reference.corpus_name} ({short(reference.corpus_hash)})",
                f"generator: {_fingerprint_name(reference, 'generator')}",
                f"judge: {_fingerprint_name(reference, 'judge')}",
                f"embedder: {_fingerprint_name(reference, 'embedder')}",
                f"api mode: {reference.api_mode}"
                + (" (latencies are those measured when recorded)" if reference.replayed else ""),
                "quality: composite of the judge axes and recall@k, weighted by `configs/gate/`",
            ]
        )
    )

    cost_points = [
        Point(summary.label, summary.projected_cost, summary.quality, False)
        for summary in summaries
    ]
    latency_points = [
        Point(summary.label, summary.p95_ms, summary.quality, False) for summary in summaries
    ]
    cost_frontier = {point.label for point in frontier(cost_points)}
    latency_frontier = {point.label for point in frontier(latency_points)}

    figures = reports_dir / FIGURES_DIR
    render(
        cost_points,
        figures / COST_FIGURE,
        "Quality against projected cost per query",
        "projected USD per query (log scale)",
        "composite quality",
        x_log=True,
    )
    render(
        latency_points,
        figures / LATENCY_FIGURE,
        "Quality against p95 serving latency",
        "p95 latency (ms)",
        "composite quality",
    )

    parts.append(heading("Results", 2))
    parts.append(
        f"`*` marks a configuration on at least one frontier "
        f"({len(cost_frontier | latency_frontier)} of {len(summaries)})."
    )
    parts.append(_results_table(summaries, cost_frontier | latency_frontier))

    measured = any(summary.measured_cost > 0 for summary in summaries)
    parts.append(heading("Frontiers", 2))
    parts.append(
        "Measured cost is zero for every run above because the generator makes no API "
        "call, so the cost frontier is plotted on projected cost: this run's context and "
        "answer tokens priced at the configured model's rates. It is a projection, not a "
        "measurement, and it is labelled as one wherever it appears."
        if not measured
        else "Cost is measured from API usage on every run below."
    )
    parts.append(
        _figure(FIGURES_DIR + "/" + COST_FIGURE, "Quality against projected cost per query")
    )
    parts.append(_figure(FIGURES_DIR + "/" + LATENCY_FIGURE, "Quality against p95 serving latency"))

    parts.append(heading("Non-dominated configurations", 3))
    parts.append(
        table(
            ["frontier", "config", "quality", "projected $/q", "p95 ms"],
            [
                [
                    axis,
                    summary.label,
                    number(summary.quality),
                    number(summary.projected_cost, 6),
                    number(summary.p95_ms, 1),
                ]
                for axis, labels in (("cost", cost_frontier), ("latency", latency_frontier))
                for summary in sorted(summaries, key=lambda item: -item.quality)
                if summary.label in labels
            ],
        )
    )

    if excluded:
        parts.append(heading("Excluded from the table", 3))
        parts.append(
            "These runs are not comparable to the group above -- a different corpus, "
            "prompt or eval set -- so they are listed rather than mixed in."
        )
        parts.append(
            table(
                ["run", "corpus", "prompts", "eval sets"],
                [
                    [
                        summary.run_id,
                        short(summary.meta.corpus_hash),
                        short(_hash_of(summary.meta.prompt_hashes)),
                        short(_hash_of(summary.meta.evalset_hashes)),
                    ]
                    for summary in excluded
                ],
            )
        )
    return "\n\n".join(parts) + "\n"


def _figure(relative: str, caption: str) -> str:
    """A figure that follows the reader's colour scheme."""
    dark = relative.replace(".png", "_dark.png")
    return (
        f"<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{dark}">\n'
        f'  <img alt="{caption}" src="{relative}">\n'
        f"</picture>"
    )


def _fingerprint_name(meta: RunMeta, key: str) -> str:
    node = meta.fingerprint.get(key)
    if not isinstance(node, dict):
        return "n/a"
    return f"{node.get('name', '?')} / {node.get('model', node.get('name', '?'))}"
