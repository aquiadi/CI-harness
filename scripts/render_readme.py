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

from omegaconf import DictConfig
from rich.console import Console

from evalgate.config import JudgeConfig, load_config, resolve_path, typed_node
from evalgate.corpus.manifest import load_manifest_if_present
from evalgate.evalsets.judgments import JudgmentSet, load_judgment_sets
from evalgate.evalsets.schemas import AnswerExample, HumanLabel, RetrievalExample
from evalgate.evalsets.store import read_jsonl
from evalgate.gate.baseline import read_baseline
from evalgate.hashing import short
from evalgate.prompts import Prompt
from evalgate.reporting.calibration import (
    agreement_rows,
    comparison_table,
    partition_labels,
    scale_of,
    swap_changes,
)
from evalgate.reporting.markdown import number, table
from evalgate.reporting.pareto import (
    RunSummary,
    assign_labels,
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
    labels = assign_labels(summaries)
    cost = {
        point.label
        for point in frontier(
            [
                Point(labels[item.run_id], item.projected_cost, item.quality, False)
                for item in summaries
            ]
        )
    }
    latency = {
        point.label
        for point in frontier(
            [Point(labels[item.run_id], item.p95_ms, item.quality, False) for item in summaries]
        )
    }
    ranked = sorted(summaries, key=lambda item: -item.quality)[:limit]
    return table(
        ["", "config", "quality", "recall@k", "nDCG@10", "p95 ms", "projected $/q"],
        [
            [
                "*" if labels[item.run_id] in cost | latency else "",
                labels[item.run_id],
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
    sets: list[JudgmentSet], labels: list[HumanLabel], fallback: JudgeConfig
) -> tuple[str, str, int, str, str]:
    """Agreement table, label source, pairs covered, one-line summary, and the judge.

    The judge is the one whose scores these are -- the first set on disk, LLM
    judges before rules -- never the configured one. Naming the configured
    judge is how the README once attributed an LLM judge's kappa to the
    rule-based baseline.
    """
    label_sets = partition_labels(labels)
    if not label_sets or not sets:
        return ("No judgments or labels on disk yet.", "none", 0, "not yet measured", "none")
    label_set = label_sets[0]
    judgment_set = sets[0]
    agreement, _ = agreement_rows(label_set, judgment_set, scale_of(judgment_set, fallback))
    rows = []
    headline: list[str] = []
    covered = 0
    for result in agreement:
        covered = max(covered, result.n)
        headline.append(
            f"{number(result.kappa) if result.is_defined else UNDEFINED} "
            f"{result.axis.replace('_', ' ')}"
        )
        rows.append(
            [
                result.axis,
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
    return (
        table(headers, rows),
        label_set.name,
        covered,
        " / ".join(headline),
        judgment_set.judge_model,
    )


def _key(name: str) -> str:
    """A template identifier from a judge model or an axis name."""
    return "".join(char if char.isalnum() else "_" for char in name).strip("_")


def _calibration_facts(
    sets: list[JudgmentSet], labels: list[HumanLabel], fallback: JudgeConfig
) -> dict[str, str]:
    """The individual numbers the calibration prose quotes, so none is typed by hand."""
    label_sets = partition_labels(labels)
    facts: dict[str, str] = {"judge_comparison": "", "swap_summary": "no swapped scorings"}
    if not label_sets or not sets:
        return facts
    label_set = label_sets[0]
    if len(sets) > 1:
        facts["judge_comparison"] = comparison_table(sets, label_set, fallback)
    for judgment_set in sets:
        scale = scale_of(judgment_set, fallback)
        agreement, _ = agreement_rows(label_set, judgment_set, scale)
        prefix = _key(judgment_set.judge_model)
        for row in agreement:
            kappa = number(row.kappa) if row.is_defined else UNDEFINED
            facts[f"kappa_{prefix}_{row.axis}"] = kappa
        if judgment_set is sets[0]:
            for row in agreement:
                facts[f"kappa_{row.axis}"] = number(row.kappa) if row.is_defined else UNDEFINED
                facts[f"qkappa_{row.axis}"] = (
                    number(row.quadratic_kappa) if row.is_defined else UNDEFINED
                )
                facts[f"bias_{row.axis}"] = f"{row.mean_judge - row.mean_human:+.3f}"
                facts[f"gap_{row.axis}"] = number(abs(row.mean_judge - row.mean_human), 2)
            moved = swap_changes(judgment_set.judgments, scale.axes)
            phrases = [
                f"{axis.replace('_', ' ')} on {changed} of {total}"
                for axis, (changed, total) in moved.items()
                if changed
            ]
            total = max((pairs for _, pairs in moved.values()), default=0)
            facts["swap_summary"] = (
                "changed " + ", ".join(phrases) + " answers and no other score"
                if phrases
                else f"changed no score on any of {total} answers"
            )
    return facts


def _sweep_facts(summaries: list[RunSummary], n_retrieval: int) -> dict[str, str]:
    """Ratios the sweep prose quotes."""
    context = [
        float(item.metrics["context_tokens_per_query"])
        for item in summaries
        if item.metrics.get("context_tokens_per_query")
    ]

    def mean_context(k: int) -> float:
        values = [
            float(item.metrics["context_tokens_per_query"])
            for item in summaries
            if item.metrics.get("k") == k and item.metrics.get("context_tokens_per_query")
        ]
        return sum(values) / len(values) if values else float("nan")

    return {
        "context_spread": f"{max(context) / min(context):.1f}" if context else UNDEFINED,
        "context_k10_over_k3": f"{mean_context(10) / mean_context(3):.1f}",
        "recall_step": f"{100.0 / n_retrieval:.1f}" if n_retrieval else UNDEFINED,
    }


REAL_PROFILE = ["+experiment=real"]


def _real_values(root: Path) -> dict[str, str]:
    """What the README says about the real regulation, from its own artifacts."""
    cfg = load_config(overrides=REAL_PROFILE)
    baseline_path = resolve_path(cfg, "gate.baseline_path")
    summaries, _ = partition_comparable(
        [
            summary
            for summary in latest_per_config(load_summaries(root / "runs"))
            if summary.meta.corpus_name == str(cfg.corpus.name)
        ]
    )
    retrieval = read_jsonl(resolve_path(cfg, "evalsets.retrieval_path"), RetrievalExample)
    answers = read_jsonl(resolve_path(cfg, "evalsets.answers_path"), AnswerExample)
    manifest = load_manifest_if_present(resolve_path(cfg, "paths.corpus_manifest_path"))
    judge = typed_node(cfg, "judge", JudgeConfig)
    sets = load_judgment_sets(resolve_path(cfg, "evalsets.judgments_dir"))
    labels = read_jsonl(resolve_path(cfg, "evalsets.human_labels_path"), HumanLabel)
    label_sets = partition_labels(labels)
    # The same per-judge facts the synthetic section quotes, prefixed, so the
    # real-corpus prose can cite a kappa without typing it.
    facts = {
        f"real_{key}": value
        for key, value in _calibration_facts(sets, labels, judge).items()
        if key != "judge_comparison"
    }
    values = {
        **facts,
        "real_documents": str(len(manifest.ok_entries)) if manifest else "0",
        "real_corpus_hash": short(manifest.corpus_hash()) if manifest else "none",
        "real_n_configs": str(len(summaries)),
        "real_pareto_table": _pareto_rows(summaries, TOP_ROWS)
        if summaries
        else "No runs over the real corpus yet.",
        "real_retrieval_slots": str(len(retrieval)),
        "real_answer_slots": str(len(answers)),
        "real_label_source": label_sets[0].name if label_sets else "none",
        "real_judge_comparison": comparison_table(sets, label_sets[0], judge)
        if sets and label_sets
        else "No judgments over the real corpus yet.",
    }
    if baseline_path.is_file():
        frozen = read_baseline(baseline_path)
        values.update(
            {
                "real_quality": number(frozen.metrics.get("composite_quality")),
                "real_recall": number(frozen.metrics.get("recall_at_k")),
                "real_p95_ms": number(float(frozen.metrics.get("p95_latency_s", 0.0)) * MS, 1),
                "real_baseline_config": "/".join(
                    str(frozen.fingerprint.get(node, {}).get("name", "?"))
                    for node in ("chunker", "retriever")
                )
                + f"/k={frozen.metrics.get('k', '?')}",
            }
        )
    return values


NOT_FROZEN = "not frozen"
FROZEN_KEYS = (
    "quality",
    "recall",
    "p95_ms",
    "grounded",
    "relevant",
    "citations",
    "generator",
    "judge",
    "config",
    "n_answers",
)


def _frozen_values(overrides: list[str], prefix: str) -> dict[str, str]:
    """The headline of another profile's frozen baseline, or 'not frozen' for each value."""
    cfg = load_config(overrides=overrides)
    path = resolve_path(cfg, "gate.baseline_path")
    if not path.is_file():
        return {f"{prefix}_{key}": NOT_FROZEN for key in FROZEN_KEYS}
    frozen = read_baseline(path)
    metrics, fingerprint = frozen.metrics, frozen.fingerprint
    return {
        f"{prefix}_quality": number(metrics.get("composite_quality")),
        f"{prefix}_recall": number(metrics.get("recall_at_k")),
        f"{prefix}_p95_ms": number(float(metrics.get("p95_latency_s", 0.0)) * MS, 1),
        f"{prefix}_grounded": number(metrics.get("judge_groundedness"), 2),
        f"{prefix}_relevant": number(metrics.get("judge_relevance"), 2),
        f"{prefix}_citations": number(metrics.get("judge_citation_correctness"), 2),
        f"{prefix}_generator": str(fingerprint.get("generator", {}).get("model", "?")),
        f"{prefix}_judge": str(fingerprint.get("judge", {}).get("model", "?")),
        f"{prefix}_config": "/".join(
            str(fingerprint.get(node, {}).get("name", "?")) for node in ("chunker", "retriever")
        )
        + f"/k={metrics.get('k', '?')}",
        f"{prefix}_n_answers": str(metrics.get("n_answers", "?")),
    }


LLM_PROFILE = ["+experiment=llm_replay"]
LIVE_GROQ_PROFILE = ["+experiment=live_groq"]


def _cassette_count(cfg: DictConfig) -> str:
    """How many recorded calls the replay profiles can serve."""
    directory = resolve_path(cfg, "paths.cassette_dir")
    return str(len(list(directory.glob("*.json")))) if directory.is_dir() else "0"


DEFAULT_OVERRIDES = ["+experiment=baseline"]


def build_values(root: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Every value the template can reference.

    Takes the same hydra overrides as the rest of the pipeline, so a README
    rendered after freezing a live baseline reports the live numbers.
    """
    cfg = load_config(overrides=DEFAULT_OVERRIDES if overrides is None else overrides)
    baseline = read_baseline(resolve_path(cfg, "gate.baseline_path"))
    metrics = baseline.metrics
    judge = typed_node(cfg, "judge", JudgeConfig)

    # The baseline's corpus only: the synthetic and real sweeps are different
    # ground and each has its own report.
    summaries, excluded = partition_comparable(
        [
            summary
            for summary in latest_per_config(load_summaries(root / "runs"))
            if summary.meta.corpus_name == baseline.corpus_name
        ]
    )
    judgment_sets = load_judgment_sets(resolve_path(cfg, "evalsets.judgments_dir"))
    labels = [
        *read_jsonl(resolve_path(cfg, "evalsets.human_labels_path"), HumanLabel),
        *read_jsonl(resolve_path(cfg, "evalsets.seed_labels_path"), HumanLabel),
    ]
    calibration_table, label_source, label_pairs, headline, calibration_judge = _calibration_rows(
        judgment_sets, labels, judge
    )

    retrieval_set = read_jsonl(resolve_path(cfg, "evalsets.retrieval_path"), RetrievalExample)
    answer_set = read_jsonl(resolve_path(cfg, "evalsets.answers_path"), AnswerExample)
    fingerprint = baseline.fingerprint

    return {
        **_real_values(root),
        **_frozen_values(LLM_PROFILE, "llm"),
        **_frozen_values(LIVE_GROQ_PROFILE, "live_groq"),
        "n_cassettes": _cassette_count(cfg),
        **_calibration_facts(judgment_sets, labels, judge),
        **_sweep_facts(summaries, int(metrics.get("n_retrieval", 0) or 0)),
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
        "calibration_judge": calibration_judge,
        "retrieval_slots": str(len(retrieval_set)),
        "answer_slots": str(len(answer_set)),
    }


def render(root: Path, overrides: list[str] | None = None) -> str:
    """Render the template against the artifacts."""
    template = Prompt(
        name=TEMPLATE,
        path=root / TEMPLATE,
        text=(root / TEMPLATE).read_text(encoding="utf-8"),
        sha256="",
    )
    return template.render(**build_values(root, overrides))


def main(argv: list[str] | None = None) -> int:
    """Write README.md, reporting whether anything changed."""
    overrides = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parent.parent
    content = render(root, overrides or None)
    output = root / OUTPUT
    changed = not output.is_file() or output.read_text(encoding="utf-8") != content
    if changed:
        output.write_text(content, encoding="utf-8")
    console.print(f"{'wrote' if changed else 'unchanged'} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
