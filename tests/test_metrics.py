from __future__ import annotations

import pytest

from evalgate.evaluation.metrics import (
    composite_quality,
    mean,
    ndcg_at_k,
    normalise_score,
    percentile,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_counts_a_hit_inside_k() -> None:
    assert recall_at_k(2, 3) == 1.0
    assert recall_at_k(3, 3) == 0.0
    assert recall_at_k(None, 10) == 0.0


def test_reciprocal_rank_is_one_based() -> None:
    assert reciprocal_rank(0) == 1.0
    assert reciprocal_rank(1) == 0.5
    assert reciprocal_rank(None) == 0.0


def test_ndcg_discounts_later_hits() -> None:
    assert ndcg_at_k(0, 10) == 1.0
    assert ndcg_at_k(1, 10) == pytest.approx(0.6309, abs=1e-4)
    assert ndcg_at_k(0, 10) > ndcg_at_k(3, 10) > 0.0


def test_ndcg_is_zero_beyond_k() -> None:
    assert ndcg_at_k(10, 10) == 0.0
    assert ndcg_at_k(None, 10) == 0.0


def test_percentile_and_mean_handle_an_empty_sample() -> None:
    assert percentile([], 95) == 0.0
    assert mean([]) == 0.0


def test_p95_is_above_p50() -> None:
    values = [float(index) for index in range(100)]
    assert percentile(values, 95) > percentile(values, 50)


def test_normalise_maps_the_scale_onto_zero_one() -> None:
    assert normalise_score(1, 1, 5) == 0.0
    assert normalise_score(5, 1, 5) == 1.0
    assert normalise_score(3, 1, 5) == 0.5


def test_normalise_rejects_a_degenerate_scale() -> None:
    with pytest.raises(ValueError, match="must increase"):
        normalise_score(3, 5, 5)


def test_composite_is_a_weighted_mean() -> None:
    assert composite_quality({"a": 1.0, "b": 0.0}, {"a": 0.75, "b": 0.25}) == 0.75


def test_composite_refuses_weights_that_do_not_sum_to_one() -> None:
    """A composite whose parts changed is not comparable to the baseline."""
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        composite_quality({"a": 1.0}, {"a": 0.9})


def test_composite_refuses_a_missing_measurement() -> None:
    with pytest.raises(ValueError, match="no measurement for weighted"):
        composite_quality({"a": 1.0}, {"a": 0.5, "b": 0.5})


def test_composite_refuses_empty_weights() -> None:
    with pytest.raises(ValueError, match="needs weights"):
        composite_quality({"a": 1.0}, {})
