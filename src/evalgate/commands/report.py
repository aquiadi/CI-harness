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
from evalgate.reporting.calibration import build_report, write_report

console = Console()

CALIBRATION_FILE = "judge_calibration.md"


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
    content = build_report(cfg, stack.tokenizer, summary)
    path = reports_dir / CALIBRATION_FILE
    changed = write_report(path, content)
    console.print(f"{'wrote' if changed else 'unchanged'} {path}")
    return 0
