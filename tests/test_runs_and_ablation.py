from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evalgate.commands.ablate import Cell, expand
from evalgate.config import AblationConfig
from evalgate.reporting.pareto import RunSummary, assign_labels, partition_comparable
from evalgate.reporting.plots import Point, frontier
from evalgate.runs.store import (
    RunExistsError,
    RunMeta,
    RunNotFoundError,
    find_measured,
    list_runs,
    load_meta,
    load_metrics,
    load_rows,
    make_run_id,
    measurement_key,
    resolve_run,
    write_run,
)

FINGERPRINT: dict[str, object] = {
    "chunker": {"name": "fixed_token"},
    "retriever": {"name": "bm25", "k": 3},
    "generator": {"name": "extractive", "model": "extractive-v1"},
    "judge": {"name": "heuristic", "model": "rule-based-v1"},
}


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


def test_a_different_judge_is_different_ground() -> None:
    """A rule's quality score and an LLM's are different units."""
    rule = meta(fingerprint={"judge": {"name": "heuristic", "model": "rule-based-v1"}})
    model = meta(fingerprint={"judge": {"name": "api", "model": "qwen/qwen3.8-27b"}})
    assert not rule.comparable_to(model)
    assert rule.comparable_to(meta("r-2", fingerprint=dict(rule.fingerprint)))


def test_runs_scored_by_another_judge_are_not_ranked_with_the_rest(tmp_path: Path) -> None:
    rule = {"judge": {"name": "heuristic", "model": "rule-based-v1"}}
    model = {"judge": {"name": "api", "model": "qwen/qwen3.8-27b"}}
    comparable, excluded = partition_comparable(
        [
            summary("a", fingerprint=rule),
            summary("b", fingerprint=rule),
            summary("c", fingerprint=model),
        ]
    )
    assert {item.run_id for item in comparable} == {"a", "b"}
    assert [item.run_id for item in excluded] == ["c"]


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


def summary_with(
    run_id: str, chunker: str, retriever: str, embedder: str, **extra: object
) -> RunSummary:
    fingerprint: dict[str, object] = {
        "chunker": {"name": chunker},
        "retriever": {"name": retriever},
        "embedder": {"name": embedder, "model": f"{embedder}-model"},
        "generator": {"name": "extractive", "model": "extractive-v1"},
        "judge": {"name": "heuristic", "model": "rule-based-v1"},
    }
    fingerprint.update(extra)
    return RunSummary(
        run_id=run_id,
        meta=meta(run_id, fingerprint=fingerprint, config_hash=run_id.ljust(64, "0")),
        metrics={"k": 10},
    )


def test_labels_stay_short_when_nothing_collides() -> None:
    """The common case -- one embedder -- keeps the short, readable label."""
    summaries = [
        summary_with("a", "fixed_token", "hybrid", "local"),
        summary_with("b", "section_aware", "bm25", "local"),
    ]
    assert set(assign_labels(summaries).values()) == {
        "fixed/hybrid/k=10",
        "section/bm25/k=10",
    }


def test_the_same_sweep_with_a_different_embedder_gets_distinct_labels() -> None:
    """Re-running a sweep with real embeddings must not produce duplicate rows."""
    summaries = [
        summary_with("a", "fixed_token", "hybrid", "hashed"),
        summary_with("b", "fixed_token", "hybrid", "local"),
    ]
    labels = assign_labels(summaries)
    assert labels["a"] != labels["b"]
    assert labels == {"a": "fixed/hybrid/k=10/hashed", "b": "fixed/hybrid/k=10/local"}


def test_only_the_colliding_rows_are_qualified() -> None:
    summaries = [
        summary_with("a", "fixed_token", "hybrid", "hashed"),
        summary_with("b", "fixed_token", "hybrid", "local"),
        summary_with("c", "section_aware", "bm25", "local"),
    ]
    assert assign_labels(summaries)["c"] == "section/bm25/k=10"


def test_a_qualifier_that_does_not_distinguish_is_not_appended() -> None:
    """Two runs differing only in generator should not gain a useless embedder tag."""
    summaries = [
        summary_with("a", "fixed_token", "hybrid", "local"),
        summary_with("b", "fixed_token", "hybrid", "local"),
    ]
    summaries[1].meta.fingerprint["generator"] = {"name": "anthropic", "model": "claude-sonnet-4-6"}
    labels = assign_labels(summaries)
    assert labels["a"] == "fixed/hybrid/k=10/extractive-v1"
    assert labels["b"] == "fixed/hybrid/k=10/claude-sonnet-4-6"


def test_indistinguishable_runs_fall_back_to_the_config_hash() -> None:
    """Every row must be identifiable, even when no named component differs."""
    summaries = [
        summary_with("aaa", "fixed_token", "hybrid", "local"),
        summary_with("bbb", "fixed_token", "hybrid", "local"),
    ]
    labels = assign_labels(summaries)
    assert labels["aaa"] != labels["bbb"]
    assert all("#" in label for label in labels.values())


def test_labels_are_unique_by_construction() -> None:
    summaries = [
        summary_with(letter, "fixed_token", "hybrid", embedder)
        for letter, embedder in zip("abcd", ["hashed", "local", "api", "hashed"], strict=True)
    ]
    labels = assign_labels(summaries)
    assert len(set(labels.values())) == len(summaries)


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
    assert len(set(assign_labels(comparable).values())) == len(comparable)


def test_re_measuring_a_configuration_shows_one_row(repo_root: Path) -> None:
    """Freezing a baseline re-measures the winning cell; the table keeps one row."""
    from evalgate.reporting.pareto import latest_per_config, load_summaries

    everything = load_summaries(repo_root / "runs")
    deduplicated = latest_per_config(everything)
    assert len(deduplicated) < len(everything)
    assert len({item.meta.config_hash for item in deduplicated}) == len(deduplicated)


def test_a_measured_cell_is_found_by_what_it_measured(tmp_path: Path) -> None:
    """The lookup that lets a sweep skip a cell before paying for it."""
    runs = tmp_path / "runs"
    recorded = meta("r-1", fingerprint=FINGERPRINT)
    write_run(runs, recorded, rows(), {"composite_quality": 0.5})

    assert find_measured(runs, measurement_key(recorded)) is not None
    for changed, why in (
        (meta(corpus_hash="z" * 64, fingerprint=FINGERPRINT), "a new corpus must re-measure"),
        (meta(evalset_hashes={"answers": "z" * 64}, fingerprint=FINGERPRINT), "new questions"),
        (meta(prompt_hashes={"generator": "z" * 64}, fingerprint=FINGERPRINT), "a new prompt"),
        (meta(fingerprint={**FINGERPRINT, "retriever": {"name": "bm25", "k": 5}}), "new k"),
    ):
        assert find_measured(runs, measurement_key(changed)) is None, why
    assert find_measured(tmp_path / "nothing-here", measurement_key(recorded)) is None


def test_a_config_hash_change_that_measures_nothing_new_is_not_re_measured(
    tmp_path: Path,
) -> None:
    """A schema field the components ignore changes the config hash and nothing else."""
    runs = tmp_path / "runs"
    write_run(runs, meta("r-1", fingerprint=FINGERPRINT), rows(), {"q": 0.5})
    renamed = meta("r-2", config_hash="d" * 64, fingerprint=FINGERPRINT)
    assert find_measured(runs, measurement_key(renamed)) is not None


def test_bm25_under_two_embedders_is_one_measurement() -> None:
    """BM25 ignores the embedder; the committed sweep measured it twice anyway."""
    hashed = meta("r-1", fingerprint={**FINGERPRINT, "embedder": {"name": "hashed"}})
    local = meta("r-2", fingerprint={**FINGERPRINT, "embedder": {"name": "local"}})
    assert measurement_key(hashed) == measurement_key(local)


def test_write_run_cannot_detect_a_duplicate_configuration(tmp_path: Path) -> None:
    """Why find_measured has to exist at all.

    A run id embeds a timestamp, so two runs of one configuration never collide
    on disk and `write_run`'s refusal to overwrite never fires. The sweep's
    duplicate check therefore ran only after `evaluate()` had already spent the
    API calls, and re-running a sweep re-paid for every cell to discard the
    result as a duplicate.
    """
    runs = tmp_path / "runs"
    first = write_run(runs, meta("20260101T000000Z-cccccccccccc"), rows(), {"q": 0.5})
    second = write_run(runs, meta("20270101T000000Z-cccccccccccc"), rows(), {"q": 0.5})

    assert first != second, "same config hash, two directories, no refusal"
    assert find_measured(runs, measurement_key(meta())) is not None


def test_a_run_id_carries_a_timestamp_so_two_runs_never_collide() -> None:
    """The mechanism behind the test above, stated directly."""
    assert make_run_id("c" * 64, "2026-01-01T00:00:00Z") != make_run_id(
        "c" * 64, "2027-01-01T00:00:00Z"
    )
