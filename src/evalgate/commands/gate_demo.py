"""``evalgate gate_demo``: prove the gate actually blocks a regression.

A gate nobody has watched fail is a gate nobody should trust. This command
runs the suite twice -- once with the retriever deliberately degraded, once
with the baseline configuration -- and asserts that the first fails and the
second passes. It exits nonzero if either expectation is wrong, which makes it
a test of the gate rather than a demonstration of it.

The degradation is a config override (`retriever.k=1` by default), so nothing
about the gate, the baseline or the eval sets is touched.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from evalgate.commands.evaluate import run_evaluation
from evalgate.config import GateConfig, load_config, resolve_path, typed_node
from evalgate.gate.baseline import read_baseline
from evalgate.gate.check import GateResult, evaluate_gate

console = Console()

DEGRADE_PREFIX = "degrade="
DEFAULT_DEGRADATION = "retriever.k=1"


@dataclass(frozen=True, slots=True)
class Step:
    """One half of the demonstration."""

    label: str
    overrides: list[str]
    expect_pass: bool


def _run(overrides: list[str]) -> GateResult:
    cfg = load_config(overrides=overrides)
    baseline = read_baseline(resolve_path(cfg, "gate.baseline_path"))
    outcome = run_evaluation(cfg)
    return evaluate_gate(
        baseline, outcome.meta, outcome.metrics, typed_node(cfg, "gate", GateConfig)
    )


def run(overrides: list[str]) -> int:
    """Degrade, expect failure; restore, expect success."""
    degradation = next(
        (arg[len(DEGRADE_PREFIX) :] for arg in overrides if arg.startswith(DEGRADE_PREFIX)),
        DEFAULT_DEGRADATION,
    )
    base = [arg for arg in overrides if not arg.startswith(DEGRADE_PREFIX)]

    steps = [
        Step(f"degraded ({degradation})", [*base, degradation], expect_pass=False),
        Step("baseline configuration", base, expect_pass=True),
    ]

    ok = True
    for step in steps:
        result = _run(step.overrides)
        matched = result.passed == step.expect_pass
        ok = ok and matched
        verdict = "passed" if result.passed else "failed"
        expected = "pass" if step.expect_pass else "fail"
        colour = "green" if matched else "red"
        console.print(f"[{colour}]{step.label}: gate {verdict}[/{colour}] (expected {expected})")
        for check in result.checks:
            delta = "-" if check.delta_pct is None else f"{check.delta_pct:+.2f}%"
            limit = f"limit {check.threshold_pct:.2f}%"
            console.print(f"    {check.name:<15} {delta:>10}  {limit}  {check.status}")
        for reason in result.blocking:
            console.print(f"    blocking: {reason}")

    if ok:
        console.print(
            "\n[green]the gate blocks a degraded retriever and passes the baseline[/green]"
        )
        return 0
    console.print("\n[red]the gate did not behave as expected[/red]")
    return 1
