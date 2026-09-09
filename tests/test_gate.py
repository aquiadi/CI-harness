from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalgate.config import GateConfig
from evalgate.gate.baseline import Baseline, BaselineError, freeze, read_baseline, write_baseline
from evalgate.gate.check import Direction, evaluate_gate
from evalgate.reporting.gate import render, write_json
from evalgate.runs.store import RunMeta

METRICS: dict[str, Any] = {
    "composite_quality": 0.800,
    "p95_latency_s": 0.010,
    "cost_per_query_usd": 0.0,
    "projected_cost_per_query_usd": 0.010,
    "recall_at_k": 0.85,
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
    assert len(payload["checks"]) == 3
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
