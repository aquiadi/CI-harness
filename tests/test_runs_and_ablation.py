from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evalgate.commands.ablate import Cell, expand
from evalgate.config import AblationConfig
from evalgate.reporting.pareto import RunSummary, partition_comparable
from evalgate.reporting.plots import Point, frontier
from evalgate.runs.store import (
    RunExistsError,
    RunMeta,
    RunNotFoundError,
    list_runs,
    load_meta,
    load_metrics,
    load_rows,
    make_run_id,
    resolve_run,
    write_run,
)


def meta(run_id: str = "r-1", **overrides: object) -> RunMeta:
    base: dict[str, object] = {
        "run_id": run_id,
        "created_at": "2026-01-01T00:00:00Z",
        "config_hash": "c" * 64,
        "corpus_name": "cbam_synthetic",
        "corpus_hash": "a" * 64,
        "index_hash": "b" * 64,
        "prompt_hashes": {"generator": "p" * 64},
        "evalset_hashes": {"answers": "e" * 64},
        "api_mode": "replay",
        "replayed": True,
        "fingerprint": {},
    }
    base.update(overrides)
    return RunMeta.model_validate(base)


def rows() -> pd.DataFrame:
    return pd.DataFrame([{"task": "answer", "example_id": "e-1", "latency_s": 0.1}])


def test_run_round_trip(tmp_path: Path) -> None:
    path = write_run(tmp_path, meta(), rows(), {"composite_quality": 0.8})
    assert load_meta(path) == meta()
    assert load_metrics(path)["composite_quality"] == 0.8
    assert len(load_rows(path)) == 1


def test_runs_are_immutable(tmp_path: Path) -> None:
    """A run whose contents can change is not evidence."""
    write_run(tmp_path, meta(), rows(), {})
    with pytest.raises(RunExistsError, match="immutable"):
        write_run(tmp_path, meta(), rows(), {})


def test_run_id_sorts_by_time_and_names_its_config() -> None:
    run_id = make_run_id("abc123" + "0" * 58, "2026-09-09T01:02:03Z")
    assert run_id.startswith("20260909T010203Z-")
    assert run_id.endswith("abc123000000")


def test_list_and_resolve_runs(tmp_path: Path) -> None:
    write_run(tmp_path, meta("20260101T000000Z-aaa"), rows(), {})
    write_run(tmp_path, meta("20260202T000000Z-bbb"), rows(), {})
    assert [path.name for path in list_runs(tmp_path)] == [
        "20260101T000000Z-aaa",
        "20260202T000000Z-bbb",
    ]
    assert resolve_run(tmp_path, None).name == "20260202T000000Z-bbb"
    assert resolve_run(tmp_path, "20260101T000000Z-aaa").name == "20260101T000000Z-aaa"


def test_resolving_an_unknown_run_raises(tmp_path: Path) -> None:
    write_run(tmp_path, meta("20260101T000000Z-aaa"), rows(), {})
    with pytest.raises(RunNotFoundError, match="no run"):
        resolve_run(tmp_path, "nope")


def test_comparability_requires_matching_upstream_hashes() -> None:
    base = meta()
    assert base.comparable_to(meta(created_at="2030-01-01T00:00:00Z"))
    assert not base.comparable_to(meta(corpus_hash="z" * 64))
    assert not base.comparable_to(meta(prompt_hashes={"generator": "z" * 64}))
    assert not base.comparable_to(meta(evalset_hashes={"answers": "z" * 64}))
    # The index hash is not part of the ground: changing the chunker changes it,
    # and comparing chunkers is what the sweep is for.
    assert base.comparable_to(meta(index_hash="z" * 64))


def summary(run_id: str, **overrides: object) -> RunSummary:
    return RunSummary(run_id=run_id, meta=meta(run_id, **overrides), metrics={"k": 3})


def test_incomparable_runs_are_excluded_not_footnoted() -> None:
    comparable, excluded = partition_comparable(
        [summary("a"), summary("b"), summary("c", corpus_hash="z" * 64)]
    )
    assert {item.run_id for item in comparable} == {"a", "b"}
    assert [item.run_id for item in excluded] == ["c"]


def test_no_runs_partitions_to_nothing() -> None:
    assert partition_comparable([]) == ([], [])


def test_frontier_keeps_the_non_dominated_set() -> None:
    points = [
        Point("cheap-poor", 1.0, 0.5, False),
        Point("cheap-good", 1.0, 0.9, False),
        Point("dear-good", 5.0, 0.9, False),
        Point("dear-best", 5.0, 0.95, False),
    ]
    labels = {point.label for point in frontier(points)}
    assert labels == {"cheap-good", "dear-best"}


def test_frontier_keeps_exact_ties() -> None:
    points = [Point("a", 1.0, 0.9, False), Point("b", 1.0, 0.9, False)]
    assert len(frontier(points)) == 2


def test_frontier_of_one_point_is_that_point() -> None:
    assert [point.label for point in frontier([Point("only", 1.0, 0.5, False)])] == ["only"]


def matrix(**overrides: object) -> AblationConfig:
    base = AblationConfig(
        chunkers=["fixed_token", "section_aware"],
        retrievers=["dense", "bm25", "hybrid"],
        k_values=[3, 10],
        rerank=[False, True],
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_sweep_covers_the_cross_product_of_meaningful_cells() -> None:
    cells, _ = expand(matrix())
    assert len(cells) == 2 * 3 * 2 + 2 * 1 * 2  # no-rerank cells, plus rerank for hybrid only
    assert all(cell.retriever == "hybrid" for cell in cells if cell.rerank)


def test_meaningless_cells_are_reported_not_silently_dropped() -> None:
    _, skipped = expand(matrix())
    assert skipped
    assert all("no rerank variant" in reason for _, reason in skipped)
    assert all(cell.retriever in {"dense", "bm25"} for cell, _ in skipped)


def test_cell_overrides_select_the_rerank_variant() -> None:
    assert "retriever=hybrid_rerank" in Cell("fixed_token", "hybrid", 5, True).overrides
    assert "retriever=hybrid" in Cell("fixed_token", "hybrid", 5, False).overrides


def test_cell_label_is_readable() -> None:
    assert str(Cell("fixed_token", "hybrid", 5, True)) == "fixed_token/hybrid+rerank/k=5"


def test_committed_sweep_has_at_least_twelve_comparable_configurations(repo_root: Path) -> None:
    """The M4 gate: the frontier is plotted from a real matrix, not two points."""
    from evalgate.reporting.pareto import latest_per_config, load_summaries

    comparable, _ = partition_comparable(latest_per_config(load_summaries(repo_root / "runs")))
    assert len(comparable) >= 12
    assert len({item.label for item in comparable}) == len(comparable)


def test_re_measuring_a_configuration_shows_one_row(repo_root: Path) -> None:
    """Freezing a baseline re-measures the winning cell; the table keeps one row."""
    from evalgate.reporting.pareto import latest_per_config, load_summaries

    everything = load_summaries(repo_root / "runs")
    deduplicated = latest_per_config(everything)
    assert len(deduplicated) < len(everything)
    assert len({item.meta.config_hash for item in deduplicated}) == len(deduplicated)
