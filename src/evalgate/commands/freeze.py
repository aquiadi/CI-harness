"""``evalgate freeze``: write baseline.json from the current configuration.

Freezing is deliberate. It runs the suite, writes the run as a committed
artifact under `runs/`, and records its metrics, one record per question, and
the hashes of the ground it measured on. Moving a baseline is how a regression
becomes permanent, so it happens in a reviewable diff and never as a side
effect of running the gate.

`from_run=<run id>` freezes an existing committed run instead of measuring
again. It is how a baseline gains a field it did not have -- per-question
records, when they were introduced -- without its numbers moving: re-measuring
would have replaced the frozen latency with whatever this machine measured.

    make freeze
    make freeze ARGS="from_run=20260909T014505Z-4b8a0ad44f15 notes=..."
"""

from __future__ import annotations

from rich.console import Console

from evalgate.commands.evaluate import run_evaluation
from evalgate.config import JudgeConfig, load_config, resolve_path, typed_node
from evalgate.corpus.manifest import utc_now_iso
from evalgate.gate.baseline import freeze as freeze_baseline
from evalgate.gate.baseline import write_baseline
from evalgate.gate.items import item_records
from evalgate.hashing import short
from evalgate.runs.store import (
    RunExistsError,
    load_meta,
    load_metrics,
    load_rows,
    resolve_run,
    write_run,
)

console = Console()
NOTES_PREFIX = "notes="
FROM_RUN_PREFIX = "from_run="
PREFIXES = (NOTES_PREFIX, FROM_RUN_PREFIX)


def _option(overrides: list[str], prefix: str) -> str | None:
    return next((arg[len(prefix) :] for arg in overrides if arg.startswith(prefix)), None)


def run(overrides: list[str]) -> int:
    """Measure the current configuration (or reuse a run) and freeze it as the baseline."""
    notes = _option(overrides, NOTES_PREFIX)
    from_run = _option(overrides, FROM_RUN_PREFIX)
    cfg = load_config(overrides=[arg for arg in overrides if not arg.startswith(PREFIXES)])
    runs_dir = resolve_path(cfg, "paths.runs_dir")

    if from_run:
        path = resolve_run(runs_dir, from_run)
        meta, rows, metrics = load_meta(path), load_rows(path), load_metrics(path)
        console.print(f"freezing the recorded run {path.name}; nothing is re-measured")
    else:
        outcome = run_evaluation(cfg)
        meta, rows, metrics = outcome.meta, outcome.rows, outcome.metrics
        try:
            path = write_run(runs_dir, meta, rows, metrics, cfg)
            console.print(f"run written to {path}")
        except RunExistsError:
            console.print(f"[yellow]run {meta.run_id} already recorded[/yellow]")

    # The run's own judge axes, where it recorded them: a run frozen from disk
    # was judged by whatever judge it was judged by, not the configured one.
    judge = meta.judge or {}
    axes = list(judge.get("axes") or typed_node(cfg, "judge", JudgeConfig).axes)
    baseline = freeze_baseline(meta, metrics, utc_now_iso(), notes, item_records(rows, axes))
    baseline_path = resolve_path(cfg, "gate.baseline_path")
    write_baseline(baseline_path, baseline)

    console.print(
        f"baseline frozen at {baseline_path}\n"
        f"  run          {baseline.run_id}\n"
        f"  quality      {baseline.metrics.get('composite_quality', float('nan')):.4f}\n"
        f"  recall@k     {baseline.metrics.get('recall_at_k', float('nan')):.4f}\n"
        f"  p95 latency  {baseline.metrics.get('p95_latency_s', float('nan')) * 1000:.2f} ms\n"
        f"  questions    {sum(len(items) for items in baseline.items.values())} recorded\n"
        f"  corpus       {baseline.corpus_name} ({short(baseline.corpus_hash)})"
    )
    return 0
