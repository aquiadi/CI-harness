"""``evalgate report``: regenerate the reports from artifacts on disk.

Reads only what is already written -- judge scores, labels, run artifacts --
and never calls a model. A report is a pure function of its inputs, so running
this twice without changing anything rewrites nothing.
"""

from __future__ import annotations

from rich.console import Console

from evalgate.config import load_config, resolve_path
from evalgate.hashing import short
from evalgate.pipeline import build_stack
from evalgate.reporting.calibration import build_report as build_calibration
from evalgate.reporting.calibration import write_report
from evalgate.reporting.pareto import build_report as build_pareto
from evalgate.reporting.pareto import load_summaries, partition_comparable

console = Console()

CALIBRATION_FILE = "judge_calibration.md"
PARETO_FILE = "pareto.md"


def run(overrides: list[str]) -> int:
    """Write reports/judge_calibration.md."""
    cfg = load_config(overrides=overrides)
    reports_dir = resolve_path(cfg, "paths.reports_dir")

    stack = build_stack(cfg)
    summary = {
        "corpus": f"{stack.corpus.name} ({short(stack.corpus.corpus_hash)})",
        "chunker": str(cfg.chunker.name),
        "embedder": f"{cfg.embedder.name} / {cfg.embedder.model}",
        "retriever": f"{cfg.retriever.name} (k={cfg.retriever.k})",
        "index": short(stack.index.meta.index_hash),
        "api mode": str(cfg.api.mode),
    }
    calibration_path = reports_dir / CALIBRATION_FILE
    changed = write_report(calibration_path, build_calibration(cfg, stack.tokenizer, summary))
    console.print(f"{'wrote' if changed else 'unchanged'} {calibration_path}")

    summaries = load_summaries(resolve_path(cfg, "paths.runs_dir"))
    comparable, excluded = partition_comparable(summaries)
    pareto_path = reports_dir / PARETO_FILE
    changed = write_report(pareto_path, build_pareto(comparable, excluded, reports_dir))
    console.print(
        f"{'wrote' if changed else 'unchanged'} {pareto_path} "
        f"({len(comparable)} comparable runs, {len(excluded)} excluded)"
    )
    return 0
