"""``evalgate freeze``: write baseline.json from the current configuration.

Freezing is deliberate. It runs the suite, writes the run as a committed
artifact under `runs/`, and records its metrics and the hashes of the ground it
measured on. Moving a baseline is how a regression becomes permanent, so it
happens in a reviewable diff and never as a side effect of running the gate.
"""

from __future__ import annotations

from rich.console import Console

from evalgate.commands.evaluate import run_evaluation
from evalgate.config import load_config, resolve_path
from evalgate.corpus.manifest import utc_now_iso
from evalgate.gate.baseline import freeze as freeze_baseline
from evalgate.gate.baseline import write_baseline
from evalgate.hashing import short
from evalgate.runs.store import RunExistsError, write_run

console = Console()
NOTES_PREFIX = "notes="


def run(overrides: list[str]) -> int:
    """Measure the current configuration and freeze it as the baseline."""
    notes = next(
        (arg[len(NOTES_PREFIX) :] for arg in overrides if arg.startswith(NOTES_PREFIX)), None
    )
    cfg = load_config(overrides=[arg for arg in overrides if not arg.startswith(NOTES_PREFIX)])

    outcome = run_evaluation(cfg)
    runs_dir = resolve_path(cfg, "paths.runs_dir")
    try:
        path = write_run(runs_dir, outcome.meta, outcome.rows, outcome.metrics, cfg)
        console.print(f"run written to {path}")
    except RunExistsError:
        console.print(f"[yellow]run {outcome.meta.run_id} already recorded[/yellow]")

    baseline = freeze_baseline(outcome.meta, outcome.metrics, utc_now_iso(), notes)
    baseline_path = resolve_path(cfg, "gate.baseline_path")
    write_baseline(baseline_path, baseline)

    console.print(
        f"baseline frozen at {baseline_path}\n"
        f"  run          {baseline.run_id}\n"
        f"  quality      {baseline.metrics.get('composite_quality', float('nan')):.4f}\n"
        f"  recall@k     {baseline.metrics.get('recall_at_k', float('nan')):.4f}\n"
        f"  p95 latency  {baseline.metrics.get('p95_latency_s', float('nan')) * 1000:.2f} ms\n"
        f"  corpus       {baseline.corpus_name} ({short(baseline.corpus_hash)})"
    )
    return 0
