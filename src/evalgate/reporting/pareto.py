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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalgate.hashing import hash_obj, short
from evalgate.reporting.markdown import bullets, heading, number, table
from evalgate.reporting.plots import Point, frontier, render
from evalgate.runs.store import RunMeta, list_runs, load_meta, load_metrics, measurement_key

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
    def base_label(self) -> str:
        """The three things the sweep varies: chunker, retriever, k.

        Not necessarily unique on its own -- see ``assign_labels``, which is
        what tables and plots should use.
        """
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


# Appended to a label, in order, until every run in a table has a distinct one.
# Each names the fingerprint node and the field within it that reads best:
# an embedder is recognisable by its name, a generator or judge by its model.
LABEL_QUALIFIERS: tuple[tuple[str, str], ...] = (
    ("embedder", "name"),
    ("generator", "model"),
    ("judge", "model"),
)
CONFIG_HASH_CHARS = 6


def _component(meta: RunMeta, node: str, field: str) -> str:
    """One fingerprint component, for use in a label."""
    value = meta.fingerprint.get(node)
    if not isinstance(value, dict):
        return "?"
    return str(value.get(field) or value.get("name") or "?")


def _collisions(labels: Mapping[str, str]) -> list[list[str]]:
    """Run ids grouped by any label more than one of them shares."""
    grouped: dict[str, list[str]] = {}
    for run_id, label in labels.items():
        grouped.setdefault(label, []).append(run_id)
    return [run_ids for run_ids in grouped.values() if len(run_ids) > 1]


def assign_labels(summaries: Sequence[RunSummary]) -> dict[str, str]:
    """Give every run a label that is unique within the table it appears in.

    A row a reader cannot tell apart from another row is not evidence. The base
    label names what the sweep varies and stays short while that is enough;
    when a table mixes runs agreeing on all three -- the same sweep re-run with
    a real embedder, say -- the component that actually differs is appended to
    the colliding rows only. Rows that were already unambiguous keep their
    short labels, so the common case is unchanged.

    A qualifier that does not split a colliding group is skipped rather than
    appended, and a group that survives every qualifier falls back to its
    config hash, which cannot collide.
    """
    labels = {summary.run_id: summary.base_label for summary in summaries}
    by_run = {summary.run_id: summary for summary in summaries}

    for node, field in LABEL_QUALIFIERS:
        groups = _collisions(labels)
        if not groups:
            return labels
        for run_ids in groups:
            values = {run_id: _component(by_run[run_id].meta, node, field) for run_id in run_ids}
            if len(set(values.values())) == 1:
                continue
            for run_id in run_ids:
                labels[run_id] = f"{labels[run_id]}/{values[run_id]}"

    for run_ids in _collisions(labels):
        for run_id in run_ids:
            digest = short(by_run[run_id].meta.config_hash, CONFIG_HASH_CHARS)
            labels[run_id] = f"{labels[run_id]}#{digest}"
    return labels


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
    """One row per measurement: the most recent run of each.

    Re-measuring a configuration is normal -- freezing a baseline does it, and
    so does re-running a cell after a change. The runs are all kept on disk,
    but the table shows the current measurement of each rather than the same
    cell twice.

    "The same cell" is decided by what was measured (`measurement_key`), not by
    the config hash. The committed sweep ran BM25 under both embedders; BM25
    ignores the embedder, so those were one measurement taken twice with
    identical results, and they appeared as two rows.
    """
    newest: dict[str, RunSummary] = {}
    for summary in sorted(summaries, key=lambda item: item.run_id):
        newest[measurement_key(summary.meta)] = summary
    return sorted(newest.values(), key=lambda item: item.run_id)


def comparable_groups(summaries: Sequence[RunSummary]) -> list[list[RunSummary]]:
    """Runs grouped by the ground they stand on, largest group first.

    Runs are comparable only when corpus, prompts, eval sets and judge match
    (`RunMeta.comparable_to`). Ties are broken by the earliest run id so the
    order, and therefore the report, is stable.
    """
    groups: dict[tuple[str, str, str, str], list[RunSummary]] = {}
    for summary in summaries:
        key = (
            summary.meta.corpus_hash,
            _hash_of(summary.meta.prompt_hashes),
            _hash_of(summary.meta.evalset_hashes),
            _hash_of(summary.meta.judge or {}),
        )
        groups.setdefault(key, []).append(summary)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0].run_id))


def partition_comparable(
    summaries: Sequence[RunSummary],
) -> tuple[list[RunSummary], list[RunSummary]]:
    """Split runs into the largest comparable group and everything else.

    Rather than footnote a mixed table, the report tabulates the largest
    comparable group in full and gives every other group a table of its own.
    """
    groups = comparable_groups(summaries)
    if not groups:
        return [], []
    largest = groups[0]
    excluded = [summary for summary in summaries if summary not in largest]
    return largest, excluded


def _hash_of(mapping: Mapping[str, Any]) -> str:
    return hash_obj(dict(mapping))


def _results_table(
    summaries: Sequence[RunSummary], frontier_labels: set[str], labels: Mapping[str, str]
) -> str:
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
                "*" if labels[summary.run_id] in frontier_labels else "",
                labels[summary.run_id],
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
    figures_subdir: str = "",
) -> str:
    """Render the ablation report and write its figures."""
    parts: list[str] = [heading(TITLE, 1)]

    if not summaries:
        parts.append("No runs under `runs/`. Run `make ablate`.")
        return "\n\n".join(parts) + "\n"

    reference = summaries[0].meta
    replayed = any(summary.meta.replayed for summary in summaries)
    parts.append(
        bullets(
            [
                f"configurations measured: {len(summaries)}",
                f"corpus: {reference.corpus_name} ({short(reference.corpus_hash)})",
                f"judge: {_fingerprint_name(reference, 'judge')}",
                f"generator: {_distinct(summaries, 'generator')}",
                f"embedder: {_distinct(summaries, 'embedder')}",
                "api mode: "
                + ", ".join(sorted({summary.meta.api_mode for summary in summaries}))
                + (" (replayed latencies are those measured when recorded)" if replayed else ""),
                "quality: composite of the judge axes and recall@k, weighted by `configs/gate/`",
            ]
        )
    )

    labels = assign_labels(summaries)
    cost_points = [
        Point(labels[summary.run_id], summary.projected_cost, summary.quality, False)
        for summary in summaries
    ]
    latency_points = [
        Point(labels[summary.run_id], summary.p95_ms, summary.quality, False)
        for summary in summaries
    ]
    cost_frontier = {point.label for point in frontier(cost_points)}
    latency_frontier = {point.label for point in frontier(latency_points)}

    # One figures directory per corpus, so two corpora's reports never
    # overwrite each other's plots.
    figures_path = f"{FIGURES_DIR}/{figures_subdir}" if figures_subdir else FIGURES_DIR
    figures = reports_dir / figures_path
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
    parts.append(_results_table(summaries, cost_frontier | latency_frontier, labels))

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
        _figure(figures_path + "/" + COST_FIGURE, "Quality against projected cost per query")
    )
    parts.append(
        _figure(figures_path + "/" + LATENCY_FIGURE, "Quality against p95 serving latency")
    )

    parts.append(heading("Non-dominated configurations", 3))
    parts.append(
        table(
            ["frontier", "config", "quality", "projected $/q", "p95 ms"],
            [
                [
                    axis,
                    labels[summary.run_id],
                    number(summary.quality),
                    number(summary.projected_cost, 6),
                    number(summary.p95_ms, 1),
                ]
                for axis, on_frontier in (("cost", cost_frontier), ("latency", latency_frontier))
                for summary in sorted(summaries, key=lambda item: -item.quality)
                if labels[summary.run_id] in on_frontier
            ],
        )
    )

    if excluded:
        parts.append(_other_groups(reference, excluded))
    return "\n\n".join(parts) + "\n"


def _differences(reference: RunMeta, other: RunMeta) -> list[str]:
    """What separates a group from the main table, in words."""
    differences: list[str] = []
    if other.judge != reference.judge:
        differences.append(f"judge `{_fingerprint_name(other, 'judge')}`")
    if other.corpus_hash != reference.corpus_hash:
        differences.append(f"corpus {other.corpus_name} ({short(other.corpus_hash)})")
    if other.prompt_hashes != reference.prompt_hashes:
        differences.append(f"prompts {short(_hash_of(other.prompt_hashes))}")
    if other.evalset_hashes != reference.evalset_hashes:
        differences.append(f"eval sets {short(_hash_of(other.evalset_hashes))}")
    return differences


def _other_groups(reference: RunMeta, excluded: Sequence[RunSummary]) -> str:
    """Every run the main table cannot hold, grouped with the runs it can stand beside."""
    parts = [
        heading("Measured on different ground", 2),
        "These runs are not comparable to the table above -- a different judge, "
        "corpus, prompt or eval set -- so each group gets a table of its own and "
        "no frontier is drawn across groups. A quality score from one judge is not "
        "in the same units as a quality score from another.",
    ]
    for group in comparable_groups(excluded):
        meta = group[0].meta
        labels = assign_labels(group)
        parts.append(heading("Different " + ", ".join(_differences(reference, meta)), 3))
        parts.append(
            table(
                [
                    "config",
                    "generator",
                    "quality",
                    "recall@k",
                    "grounded",
                    "relevant",
                    "citations",
                    "p95 ms",
                    "projected $/q",
                ],
                [
                    [
                        labels[summary.run_id],
                        _fingerprint_name(summary.meta, "generator"),
                        number(summary.quality),
                        number(summary.metrics.get("recall_at_k")),
                        number(summary.metrics.get("judge_groundedness"), 2),
                        number(summary.metrics.get("judge_relevance"), 2),
                        number(summary.metrics.get("judge_citation_correctness"), 2),
                        number(summary.p95_ms, 1),
                        number(summary.projected_cost, 6),
                    ]
                    for summary in sorted(group, key=lambda item: -item.quality)
                ],
            )
        )
    return "\n\n".join(parts)


def _distinct(summaries: Sequence[RunSummary], key: str) -> str:
    """Every value a fingerprint node takes across the table, not just the first."""
    return ", ".join(sorted({_fingerprint_name(summary.meta, key) for summary in summaries}))


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
