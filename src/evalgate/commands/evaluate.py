"""``evalgate eval``: run the suite against the baseline and set the exit code.

This is the command that fails a build. It runs the eval sets against the
current configuration, compares the result to `baseline.json` under the
thresholds in `configs/gate/`, writes the verdict to `reports/gate.md` and
`reports/gate.json`, and exits nonzero when a threshold is exceeded, when a
metric is missing, or when the run is not comparable to the baseline.

The run itself is written under `runs/scratch/`, which is gitignored: checking
the gate should not add a committed artifact every time.
"""

from __future__ import annotations

from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from evalgate.config import GateConfig, JudgeConfig, load_config, resolve_path, typed_node
from evalgate.evaluation.runner import RunOutcome, evaluate
from evalgate.gate.baseline import read_baseline
from evalgate.gate.check import evaluate_gate, load_latency_reference
from evalgate.gate.items import compare
from evalgate.generation.factory import build_generator
from evalgate.judging.factory import build_judge
from evalgate.models.client import build_model_client
from evalgate.pipeline import build_stack
from evalgate.prompts import load_prompt
from evalgate.reporting.gate import render, write_json
from evalgate.reporting.markdown import number
from evalgate.runs.store import write_run

console = Console()

GATE_REPORT = "gate.md"
GATE_JSON = "gate.json"
PROVIDER_EXTRACTIVE = "extractive"
PROVIDER_HEURISTIC = "heuristic"


def run_evaluation(cfg: DictConfig) -> RunOutcome:
    """Build the stack and measure it. Shared by `eval` and `freeze`."""
    prompts_dir = resolve_path(cfg, "paths.prompts_dir")
    prompts = {
        "generator": load_prompt(prompts_dir, cfg.generator.prompt),
        "judge": load_prompt(prompts_dir, cfg.judge.prompt),
    }
    client = None
    if cfg.generator.provider != PROVIDER_EXTRACTIVE or cfg.judge.provider != PROVIDER_HEURISTIC:
        client = build_model_client(cfg)

    stack = build_stack(cfg)
    return evaluate(
        cfg,
        stack,
        build_generator(cfg, stack.tokenizer, client),
        build_judge(cfg, client),
        prompts,
    )


def run(overrides: list[str]) -> int:
    """Measure, compare to the baseline, and set the exit code."""
    cfg = load_config(overrides=overrides)
    gate_cfg = typed_node(cfg, "gate", GateConfig)
    judge_cfg = typed_node(cfg, "judge", JudgeConfig)
    baseline = read_baseline(resolve_path(cfg, "gate.baseline_path"))

    outcome = run_evaluation(cfg)
    write_run(
        resolve_path(cfg, "paths.scratch_runs_dir"),
        outcome.meta,
        outcome.rows,
        outcome.metrics,
        cfg,
    )
    reference = (
        load_latency_reference(resolve_path(cfg, "gate.latency_reference_path"))
        if gate_cfg.latency_reference_path
        else None
    )
    result = evaluate_gate(baseline, outcome.meta, outcome.metrics, gate_cfg, reference)
    comparison = compare(
        baseline.items,
        outcome.rows,
        judge_cfg.axes,
        gate_cfg.composite_weights,
        (judge_cfg.scale_min, judge_cfg.scale_max),
        gate_cfg.bootstrap_resamples,
        gate_cfg.confidence,
        int(cfg.seed),
    )

    table = Table(title="quality gate", show_edge=False)
    for column in ("check", "reference", "this run", "delta", "limit", ""):
        table.add_column(column, justify="right" if column not in {"check", ""} else "left")
    for check in result.checks:
        table.add_row(
            check.name,
            number(check.baseline, 6),
            number(check.current, 6),
            check.delta_label,
            check.limit_label,
            "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]",
        )
    console.print(table)
    if comparison.interval is not None:
        interval = comparison.interval
        console.print(
            f"composite delta {interval.delta:+.4f}, {interval.confidence:.0%} bootstrap "
            f"interval {interval.low:+.4f} to {interval.high:+.4f} (reported, not gated)"
        )
    for change in comparison.worse[: gate_cfg.items_listed]:
        console.print(
            f"[yellow]worse:[/yellow] {change.example_id} {change.measure} "
            f"{change.before:g} -> {change.after:g}"
        )

    reports_dir = resolve_path(cfg, "paths.reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / GATE_REPORT).write_text(
        render(
            result,
            baseline,
            outcome.meta,
            outcome.metrics,
            comparison,
            gate_cfg.items_listed,
        ),
        encoding="utf-8",
    )
    write_json(reports_dir / GATE_JSON, result, baseline, outcome.meta, outcome.metrics, comparison)

    for reason in result.blocking:
        console.print(f"[red]blocking:[/red] {reason}")
    console.print(
        "[green]gate passed[/green]" if result.passed else "[red]gate failed[/red]",
        f"-> {reports_dir / GATE_REPORT}",
    )
    return 0 if result.passed else 1
