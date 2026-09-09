#!/usr/bin/env python
"""Render README.md from README.template.md and the artifacts on disk.

    make readme

The repository's own rule is that no figure is ever typed by hand, and the
README is the most-read place for a number to quietly go stale. So it is
generated: the template holds the prose and `${placeholders}`, and every value
comes from `baseline.json`, `runs/` or the eval sets. A test asserts that the
committed README is what the current artifacts render to, so a stale number
fails the build rather than misleading a reader.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from rich.console import Console

from evalgate.config import JudgeConfig, load_config, resolve_path, typed_node
from evalgate.evalsets.schemas import AnswerExample, HumanLabel, JudgeScore, RetrievalExample
from evalgate.evalsets.store import read_jsonl
from evalgate.evaluation.agreement import axis_agreement
from evalgate.gate.baseline import read_baseline
from evalgate.hashing import short
from evalgate.prompts import Prompt
from evalgate.reporting.calibration import partition_labels
from evalgate.reporting.markdown import number, table
from evalgate.reporting.pareto import (
    RunSummary,
    latest_per_config,
    load_summaries,
    partition_comparable,
)
from evalgate.reporting.plots import Point, frontier

console = Console()

TEMPLATE = "README.template.md"
OUTPUT = "README.md"
MS = 1000.0
TOP_ROWS = 8
UNDEFINED = "undefined"


def _pareto_rows(summaries: list[RunSummary], limit: int) -> str:
    cost = {
        point.label
        for point in frontier(
            [Point(item.label, item.projected_cost, item.quality, False) for item in summaries]
        )
    }
    latency = {
        point.label
        for point in frontier(
            [Point(item.label, item.p95_ms, item.quality, False) for item in summaries]
        )
    }
    ranked = sorted(summaries, key=lambda item: -item.quality)[:limit]
    return table(
        ["", "config", "quality", "recall@k", "nDCG@10", "p95 ms", "projected $/q"],
        [
            [
                "*" if item.label in cost | latency else "",
                item.label,
                number(item.quality),
                number(item.metrics.get("recall_at_k")),
                number(item.metrics.get("ndcg_at_10")),
                number(item.p95_ms, 1),
                number(item.projected_cost, 6),
            ]
            for item in ranked
        ],
    )


def _calibration_rows(
    judgments: list[JudgeScore], labels: list[HumanLabel], judge: JudgeConfig
) -> tuple[str, str, int, str]:
    """Agreement table, the label source, the pairs covered, and a one-line summary."""
    label_sets = partition_labels(labels)
    if not label_sets or not judgments:
        return ("No judgments or labels on disk yet.", "none", 0, "not yet measured")
    label_set = label_sets[0]
    by_key = {
        (item.example_id, item.answer_hash): item for item in judgments if item.variant == "primary"
    }
    rows = []
    headline: list[str] = []
    covered = 0
    for axis in judge.axes:
        human: list[int] = []
        judged: list[int] = []
        for label in sorted(label_set.labels, key=lambda item: item.example_id):
            judgment = by_key.get((label.example_id, label.answer_hash))
            if judgment is None:
                continue
            human.append(label.scores.as_dict()[axis])
            judged.append(judgment.scores.as_dict()[axis])
        result = axis_agreement(axis, human, judged, judge.scale_min, judge.scale_max)
        covered = max(covered, result.n)
        headline.append(
            f"{number(result.kappa) if result.is_defined else UNDEFINED} {axis.replace('_', ' ')}"
        )
        rows.append(
            [
                axis,
                result.n,
                number(result.kappa) if result.is_defined else UNDEFINED,
                number(result.quadratic_kappa) if result.is_defined else UNDEFINED,
                number(result.exact_agreement),
                number(result.mean_human, 2),
                number(result.mean_judge, 2),
            ]
        )
    headers = [
        "axis",
        "n",
        "kappa",
        "quadratic kappa",
        "exact agreement",
        "mean human",
        "mean judge",
    ]
    return table(headers, rows), label_set.name, covered, " / ".join(headline)


def build_values(root: Path) -> dict[str, Any]:
    """Every value the template can reference."""
    cfg = load_config(overrides=["+experiment=baseline"])
    baseline = read_baseline(resolve_path(cfg, "gate.baseline_path"))
    metrics = baseline.metrics
    judge = typed_node(cfg, "judge", JudgeConfig)

    summaries, excluded = partition_comparable(latest_per_config(load_summaries(root / "runs")))
    judgments = read_jsonl(resolve_path(cfg, "evalsets.judge_scores_path"), JudgeScore)
    labels = [
        *read_jsonl(resolve_path(cfg, "evalsets.human_labels_path"), HumanLabel),
        *read_jsonl(resolve_path(cfg, "evalsets.seed_labels_path"), HumanLabel),
    ]
    calibration_table, label_source, label_pairs, headline = _calibration_rows(
        judgments, labels, judge
    )

    retrieval_set = read_jsonl(resolve_path(cfg, "evalsets.retrieval_path"), RetrievalExample)
    answer_set = read_jsonl(resolve_path(cfg, "evalsets.answers_path"), AnswerExample)
    fingerprint = baseline.fingerprint

    return {
        "quality": number(metrics.get("composite_quality")),
        "recall": number(metrics.get("recall_at_k")),
        "ndcg": number(metrics.get("ndcg_at_10")),
        "mrr": number(metrics.get("mrr")),
        "p50_ms": number(float(metrics.get("p50_latency_s", 0.0)) * MS, 1),
        "p95_ms": number(float(metrics.get("p95_latency_s", 0.0)) * MS, 1),
        "measured_cost": number(metrics.get("cost_per_query_usd"), 6),
        "projected_cost": number(metrics.get("projected_cost_per_query_usd"), 5),
        "context_tokens": number(metrics.get("context_tokens_per_query"), 0),
        "grounded": number(metrics.get("judge_groundedness"), 2),
        "relevant": number(metrics.get("judge_relevance"), 2),
        "citations": number(metrics.get("judge_citation_correctness"), 2),
        "baseline_run": baseline.run_id,
        "corpus": baseline.corpus_name,
        "corpus_hash": short(baseline.corpus_hash),
        "chunker": str(fingerprint.get("chunker", {}).get("name", "?")),
        "retriever": str(fingerprint.get("retriever", {}).get("name", "?")),
        "k": str(metrics.get("k", "?")),
        "embedder": str(fingerprint.get("embedder", {}).get("name", "?")),
        "generator": str(fingerprint.get("generator", {}).get("model", "?")),
        "judge_model": str(fingerprint.get("judge", {}).get("model", "?")),
        "n_retrieval_scored": str(metrics.get("n_retrieval", "?")),
        "n_answers_scored": str(metrics.get("n_answers", "?")),
        "n_configs": str(len(summaries)),
        "n_excluded": str(len(excluded)),
        "pareto_table": _pareto_rows(summaries, TOP_ROWS),
        "calibration_table": calibration_table,
        "label_source": label_source,
        "label_pairs": str(label_pairs),
        "calibration_headline": headline,
        "retrieval_slots": str(len(retrieval_set)),
        "answer_slots": str(len(answer_set)),
    }


def render(root: Path) -> str:
    """Render the template against the artifacts."""
    template = Prompt(
        name=TEMPLATE,
        path=root / TEMPLATE,
        text=(root / TEMPLATE).read_text(encoding="utf-8"),
        sha256="",
    )
    return template.render(**build_values(root))


def main(argv: list[str] | None = None) -> int:
    """Write README.md, reporting whether anything changed."""
    del argv
    root = Path(__file__).resolve().parent.parent
    content = render(root)
    output = root / OUTPUT
    changed = not output.is_file() or output.read_text(encoding="utf-8") != content
    if changed:
        output.write_text(content, encoding="utf-8")
    console.print(f"{'wrote' if changed else 'unchanged'} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
