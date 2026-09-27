"""``evalgate report``: regenerate the reports from artifacts on disk.

Reads only what is already written -- judge scores, labels, run artifacts --
and never calls a model. A report is a pure function of its inputs, so running
this twice without changing anything rewrites nothing.

One pair of reports per corpus: `reports/pareto_<corpus>.md` and
`reports/judge_calibration_<corpus>.md`. Runs over the synthetic documents and
runs over the real regulation stand on different ground and are never put in
one table; separate files make that the layout rather than a footnote.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from evalgate.config import load_config, resolve_path
from evalgate.pipeline import build_tokenizer
from evalgate.reporting.calibration import build_report as build_calibration
from evalgate.reporting.calibration import write_report
from evalgate.reporting.pareto import (
    RunSummary,
    latest_per_config,
    load_summaries,
    partition_comparable,
)
from evalgate.reporting.pareto import build_report as build_pareto
from evalgate.rootdir import config_dir

console = Console()

CALIBRATION_FILE = "judge_calibration_{corpus}.md"
PARETO_FILE = "pareto_{corpus}.md"


def corpora(eval_dir: Path, summaries: list[RunSummary]) -> list[str]:
    """Every corpus with an eval set or a run on disk."""
    names = {summary.meta.corpus_name for summary in summaries}
    if eval_dir.is_dir():
        names |= {path.name for path in eval_dir.iterdir() if path.is_dir()}
    return sorted(names)


def pareto_for(summaries: list[RunSummary], corpus: str, reports_dir: Path) -> str:
    """The ablation report for one corpus's runs."""
    comparable, excluded = partition_comparable(
        [summary for summary in summaries if summary.meta.corpus_name == corpus]
    )
    return build_pareto(comparable, excluded, reports_dir, figures_subdir=corpus)


def run(overrides: list[str]) -> int:
    """Write the pareto and calibration reports for every corpus."""
    cfg = load_config(overrides=overrides)
    reports_dir = resolve_path(cfg, "paths.reports_dir")
    summaries = latest_per_config(load_summaries(resolve_path(cfg, "paths.runs_dir")))

    for corpus in corpora(resolve_path(cfg, "paths.eval_dir"), summaries):
        # A corpus is selected by its config group, which bears its name; one
        # without a group (a test fixture) has runs but no eval set to report.
        if (config_dir() / "corpus" / f"{corpus}.yaml").is_file():
            corpus_cfg = load_config(overrides=[*overrides, f"corpus={corpus}"])
            path = reports_dir / CALIBRATION_FILE.format(corpus=corpus)
            report = build_calibration(corpus_cfg, build_tokenizer(corpus_cfg))
            changed = write_report(path, report)
            console.print(f"{'wrote' if changed else 'unchanged'} {path}")

        path = reports_dir / PARETO_FILE.format(corpus=corpus)
        changed = write_report(path, pareto_for(summaries, corpus, reports_dir))
        console.print(f"{'wrote' if changed else 'unchanged'} {path}")
    return 0
