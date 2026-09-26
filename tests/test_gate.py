from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalgate.config import GateConfig
from evalgate.gate.baseline import Baseline, BaselineError, freeze, read_baseline, write_baseline
from evalgate.gate.check import (
    Direction,
    LatencyReference,
    evaluate_gate,
    load_latency_reference,
)
from evalgate.reporting.gate import render, write_json
from evalgate.runs.store import RunMeta

METRICS: dict[str, Any] = {
    "composite_quality": 0.800,
    "p95_latency_s": 0.010,
    "cost_per_query_usd": 0.0,
    "projected_cost_per_query_usd": 0.010,
    "recall_at_k": 0.85,
    "composite_components": {"groundedness": 0.8},
}


def meta(**overrides: Any) -> RunMeta:
    base: dict[str, Any] = {
        "run_id": "20260101T000000Z-abc",
        "created_at": "2026-01-01T00:00:00Z",
        "config_hash": "c" * 64,
        "corpus_name": "cbam_synthetic",
        "corpus_hash": "a" * 64,
        "index_hash": "i" * 64,
        "prompt_hashes": {"generator": "p" * 64},
        "evalset_hashes": {"answers": "e" * 64},
        "api_mode": "replay",
        "replayed": False,
        "fingerprint": {},
    }
    base.update(overrides)
    return RunMeta.model_validate(base)


def baseline(**overrides: Any) -> Baseline:
    frozen = freeze(meta(), dict(METRICS), "2026-01-01T00:00:00Z")
    return frozen.model_copy(update=overrides)


def gate_config(**overrides: Any) -> GateConfig:
    config = GateConfig(
        baseline_path="baseline.json",
        quality_drop_pct=2.0,
        p95_latency_rise_pct=20.0,
        cost_per_query_rise_pct=15.0,
        composite_weights={"groundedness": 1.0},
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def metrics(**overrides: Any) -> dict[str, Any]:
    updated = dict(METRICS)
    updated.update(overrides)
    return updated


def test_an_unchanged_run_passes() -> None:
    result = evaluate_gate(baseline(), meta(), metrics(), gate_config())
    assert result.passed
    assert not result.blocking


def test_a_quality_drop_beyond_the_threshold_fails() -> None:
    result = evaluate_gate(baseline(), meta(), metrics(composite_quality=0.70), gate_config())
    assert not result.passed
    assert any("quality" in reason for reason in result.blocking)


def test_a_quality_drop_within_the_threshold_passes() -> None:
    result = evaluate_gate(baseline(), meta(), metrics(composite_quality=0.792), gate_config())
    assert result.passed


def test_a_quality_improvement_passes() -> None:
    assert evaluate_gate(baseline(), meta(), metrics(composite_quality=0.95), gate_config()).passed


def test_a_latency_rise_beyond_the_threshold_fails() -> None:
    result = evaluate_gate(baseline(), meta(), metrics(p95_latency_s=0.0125), gate_config())
    assert not result.passed
    assert any("p95" in reason for reason in result.blocking)


def test_a_latency_fall_passes() -> None:
    assert evaluate_gate(baseline(), meta(), metrics(p95_latency_s=0.001), gate_config()).passed


def test_a_cost_rise_beyond_the_threshold_fails() -> None:
    result = evaluate_gate(
        baseline(), meta(), metrics(projected_cost_per_query_usd=0.02), gate_config()
    )
    assert not result.passed
    assert any("cost" in reason for reason in result.blocking)


def test_cost_is_compared_on_measured_spend_when_the_baseline_had_any() -> None:
    frozen = baseline(metrics=metrics(cost_per_query_usd=0.005))
    result = evaluate_gate(frozen, meta(), metrics(cost_per_query_usd=0.005), gate_config())
    assert result.cost_metric == "cost_per_query_usd"


def test_cost_falls_back_to_the_projection_for_an_offline_baseline() -> None:
    """An offline baseline's measured cost is zero, which would make the check vacuous."""
    assert evaluate_gate(baseline(), meta(), metrics(), gate_config()).cost_metric == (
        "projected_cost_per_query_usd"
    )


def test_a_missing_metric_fails_rather_than_being_skipped() -> None:
    incomplete = {key: value for key, value in METRICS.items() if key != "composite_quality"}
    result = evaluate_gate(baseline(), meta(), incomplete, gate_config())
    assert not result.passed
    assert any("missing" in reason for reason in result.blocking)


def test_a_metric_missing_from_the_baseline_fails() -> None:
    stripped = baseline(metrics={"p95_latency_s": 0.01, "projected_cost_per_query_usd": 0.01})
    result = evaluate_gate(stripped, meta(), metrics(), gate_config())
    assert not result.passed


def test_a_changed_corpus_fails_as_incomparable() -> None:
    result = evaluate_gate(baseline(), meta(corpus_hash="z" * 64), metrics(), gate_config())
    assert not result.passed
    assert not result.comparable
    assert any("corpus hash" in reason for reason in result.blocking)


def test_a_changed_prompt_fails_as_incomparable() -> None:
    result = evaluate_gate(
        baseline(), meta(prompt_hashes={"generator": "z" * 64}), metrics(), gate_config()
    )
    assert not result.passed
    assert any("prompt generator" in reason for reason in result.blocking)


def test_a_changed_eval_set_fails_as_incomparable() -> None:
    result = evaluate_gate(
        baseline(), meta(evalset_hashes={"answers": "z" * 64}), metrics(), gate_config()
    )
    assert not result.passed
    assert any("eval set answers" in reason for reason in result.blocking)


def test_a_changed_index_is_still_comparable() -> None:
    """Changing the chunker changes the index; judging that change is the point."""
    assert evaluate_gate(baseline(), meta(index_hash="z" * 64), metrics(), gate_config()).passed


def test_thresholds_come_from_config_not_code() -> None:
    strict = gate_config(quality_drop_pct=0.0)
    assert not evaluate_gate(baseline(), meta(), metrics(composite_quality=0.799), strict).passed
    lax = gate_config(quality_drop_pct=90.0)
    assert evaluate_gate(baseline(), meta(), metrics(composite_quality=0.1), lax).passed


def test_a_zero_baseline_treats_any_increase_as_a_regression() -> None:
    frozen = baseline(metrics=metrics(p95_latency_s=0.0))
    assert not evaluate_gate(frozen, meta(), metrics(p95_latency_s=0.001), gate_config()).passed
    assert evaluate_gate(frozen, meta(), metrics(p95_latency_s=0.0), gate_config()).passed


def test_a_missing_baseline_file_is_a_failure_not_a_pass(tmp_path: Path) -> None:
    """The state where a regression is invisible must not be the state that is green."""
    with pytest.raises(BaselineError, match="no baseline"):
        read_baseline(tmp_path / "absent.json")


def test_baseline_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    original = baseline()
    write_baseline(path, original)
    assert read_baseline(path) == original


def test_report_leads_with_the_verdict() -> None:
    failing = evaluate_gate(baseline(), meta(), metrics(composite_quality=0.5), gate_config())
    report = render(failing, baseline(), meta(), metrics(composite_quality=0.5))
    assert "**FAIL**" in report
    assert "quality" in report
    passing = evaluate_gate(baseline(), meta(), metrics(), gate_config())
    assert "**PASS**" in render(passing, baseline(), meta(), metrics())


def test_report_json_is_machine_readable(tmp_path: Path) -> None:
    import json

    result = evaluate_gate(baseline(), meta(), metrics(composite_quality=0.5), gate_config())
    path = tmp_path / "gate.json"
    write_json(path, result, baseline(), meta())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert len(payload["checks"]) == 4, "three headline checks and one per component"
    assert payload["run_metrics"]["composite_quality"] is None, "no metrics were passed"
    assert payload["baseline_run_id"] == "20260101T000000Z-abc"


def test_directions_are_explicit() -> None:
    result = evaluate_gate(baseline(), meta(), metrics(), gate_config())
    directions = {check.name: check.direction for check in result.checks}
    assert directions["quality"] is Direction.HIGHER_IS_BETTER
    assert directions["p95 latency"] is Direction.LOWER_IS_BETTER
    assert directions["cost per query"] is Direction.LOWER_IS_BETTER


def test_committed_baseline_matches_the_committed_corpus(repo_root: Path) -> None:
    frozen = read_baseline(repo_root / "baseline.json")
    assert frozen.corpus_name == "cbam_synthetic"
    assert frozen.metrics["composite_quality"] > 0.0
    assert frozen.prompt_hashes
    assert frozen.evalset_hashes


def test_a_different_judge_is_not_comparable() -> None:
    """The judge is the ruler: a composite scored by another judge is in other units."""
    rule = {"judge": {"name": "heuristic", "model": "rule-based-v1"}}
    model = {"judge": {"name": "api", "model": "qwen/qwen3.8-27b"}}
    frozen = baseline(fingerprint=rule)
    result = evaluate_gate(frozen, meta(fingerprint=model), metrics(), gate_config())
    assert not result.passed
    assert not result.comparable
    assert "judge" in result.blocking[0] and "qwen/qwen3.8-27b" in result.blocking[0]


def test_the_same_judge_with_a_different_setting_is_not_comparable() -> None:
    """A different prompt or reasoning effort is a different instrument too."""
    before = {"judge": {"name": "api", "model": "m", "prompt_hash": "a"}}
    after = {"judge": {"name": "api", "model": "m", "prompt_hash": "b"}}
    result = evaluate_gate(
        baseline(fingerprint=before), meta(fingerprint=after), metrics(), gate_config()
    )
    assert not result.comparable


def test_a_changed_generator_is_still_judged_not_refused() -> None:
    """The generator is the system under test; judging its change is the point."""
    before = {"generator": {"model": "a"}, "judge": {"model": "j"}}
    after = {"generator": {"model": "b"}, "judge": {"model": "j"}}
    result = evaluate_gate(
        baseline(fingerprint=before), meta(fingerprint=after), metrics(), gate_config()
    )
    assert result.comparable


def test_a_component_collapse_fails_even_when_the_composite_holds() -> None:
    """Citation correctness falling while recall rises reads as no change at all."""
    weights = {"groundedness": 0.5, "recall_at_k": 0.5}
    frozen = baseline(
        metrics=metrics(composite_components={"groundedness": 0.8, "recall_at_k": 0.8})
    )
    run = metrics(composite_components={"groundedness": 0.6, "recall_at_k": 1.0})
    result = evaluate_gate(frozen, meta(), run, gate_config(composite_weights=weights))
    assert not result.passed
    assert any(reason.startswith("component: groundedness") for reason in result.blocking)
    quality = next(check for check in result.checks if check.name == "quality")
    assert quality.passed


def test_a_small_component_drop_passes() -> None:
    run = metrics(composite_components={"groundedness": 0.75})
    assert evaluate_gate(baseline(), meta(), run, gate_config(component_drop_max=0.1)).passed


def test_a_missing_component_fails() -> None:
    run = metrics(composite_components={})
    result = evaluate_gate(baseline(), meta(), run, gate_config())
    assert not result.passed
    assert any("missing" in reason for reason in result.blocking)


def test_latency_is_compared_to_the_reference_run_when_one_is_given() -> None:
    """CI measures the base commit on the same runner; hardware drops out."""
    slow_runner = metrics(p95_latency_s=0.050)
    reference = LatencyReference(source="base/gate.json", p95_latency_s=0.048)
    result = evaluate_gate(baseline(), meta(), slow_runner, gate_config(), reference)
    latency = next(check for check in result.checks if check.name == "p95 latency")
    assert latency.passed, "5x the baseline, but only 4% over the same runner's base commit"
    assert latency.compared_to == "the reference run (base/gate.json)"


def test_an_unreadable_latency_reference_fails() -> None:
    reference = LatencyReference(source="base/gate.json", p95_latency_s=None)
    result = evaluate_gate(baseline(), meta(), metrics(), gate_config(), reference)
    assert not result.passed
    assert any("missing" in reason and "p95" in reason for reason in result.blocking)


@pytest.mark.parametrize(
    "payload",
    [
        {"run_metrics": {"p95_latency_s": 0.02}},
        {"checks": [{"metric": "p95_latency_s", "current": 0.02}]},
        {"p95_latency_s": 0.02},
    ],
    ids=["gate.json", "older gate.json", "metrics.json"],
)
def test_a_latency_reference_is_read_from_any_run_format(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    """The base commit's code may predate the current gate.json shape."""
    import json

    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_latency_reference(path).p95_latency_s == 0.02


def test_a_missing_latency_reference_file_has_no_value(tmp_path: Path) -> None:
    assert load_latency_reference(tmp_path / "absent.json").p95_latency_s is None
