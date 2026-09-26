"""``evalgate report``: regenerate the reports from artifacts on disk.

Reads only what is already written -- judge scores, labels, run artifacts --
and never calls a model. A report is a pure function of its inputs, so running
this twice without changing anything rewrites nothing.
"""

from __future__ import annotations

from rich.console import Console

from evalgate.config import load_config, resolve_path
from evalgate.pipeline import build_tokenizer
from evalgate.reporting.calibration import build_report as build_calibration
from evalgate.reporting.calibration import write_report
from evalgate.reporting.pareto import build_report as build_pareto
from evalgate.reporting.pareto import latest_per_config, load_summaries, partition_comparable

console = Console()

CALIBRATION_FILE = "judge_calibration.md"
PARETO_FILE = "pareto.md"


def run(overrides: list[str]) -> int:
    """Write reports/judge_calibration.md and reports/pareto.md."""
    cfg = load_config(overrides=overrides)
    reports_dir = resolve_path(cfg, "paths.reports_dir")

    # No stack is built: every judgment set carries the stack it was scored
    # over, and describing it from the current config is how the report once
    # attributed one judge's scores to another.
    calibration_path = reports_dir / CALIBRATION_FILE
    changed = write_report(calibration_path, build_calibration(cfg, build_tokenizer(cfg)))
    console.print(f"{'wrote' if changed else 'unchanged'} {calibration_path}")

    summaries = latest_per_config(load_summaries(resolve_path(cfg, "paths.runs_dir")))
    comparable, excluded = partition_comparable(summaries)
    pareto_path = reports_dir / PARETO_FILE
    changed = write_report(pareto_path, build_pareto(comparable, excluded, reports_dir))
    console.print(
        f"{'wrote' if changed else 'unchanged'} {pareto_path} "
        f"({len(comparable)} comparable runs, {len(excluded)} excluded)"
    )
    return 0
