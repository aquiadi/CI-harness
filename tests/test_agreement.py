from __future__ import annotations

import pytest

from evalgate.evaluation.agreement import Disagreement, axis_agreement, worst_disagreements
from evalgate.evaluation.bias import length_bias, position_consistency, self_preference


def test_perfect_agreement_gives_kappa_one() -> None:
    result = axis_agreement("g", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1, 5)
    assert result.kappa == 1.0
    assert result.quadratic_kappa == 1.0
    assert result.exact_agreement == 1.0


def test_kappa_is_undefined_when_a_rater_never_varies() -> None:
    """A judge that always says 5 is not in chance agreement; it is unmeasurable."""
    result = axis_agreement("g", [5, 4, 3, 5], [5, 5, 5, 5], 1, 5)
    assert result.kappa is None
    assert result.quadratic_kappa is None
    assert result.undefined_reason is not None
    # It still agreed half the time, which is exactly why raw agreement alone
    # would flatter it.
    assert result.exact_agreement == 0.5


def test_empty_overlap_is_reported_not_zeroed() -> None:
    result = axis_agreement("g", [], [], 1, 5)
    assert result.n == 0
    assert result.kappa is None
    assert "no overlapping" in (result.undefined_reason or "")


def test_quadratic_kappa_punishes_distant_errors_less_than_kappa_ignores_them() -> None:
    near = axis_agreement("g", [1, 2, 3, 4, 5], [1, 2, 3, 4, 4], 1, 5)
    far = axis_agreement("g", [1, 2, 3, 4, 5], [1, 2, 3, 4, 1], 1, 5)
    assert near.quadratic_kappa is not None
    assert far.quadratic_kappa is not None
    assert near.quadratic_kappa > far.quadratic_kappa


def test_signed_error_shows_which_way_the_judge_leans() -> None:
    generous = axis_agreement("g", [3, 3, 3, 2], [4, 4, 4, 4], 1, 5)
    assert generous.mean_signed_error > 0
    harsh = axis_agreement("g", [4, 4, 4, 5], [3, 3, 3, 3], 1, 5)
    assert harsh.mean_signed_error < 0


def test_within_one_is_looser_than_exact() -> None:
    result = axis_agreement("g", [3, 3, 3, 3], [3, 4, 4, 5], 1, 5)
    assert result.exact_agreement == 0.25
    assert result.within_one == 0.75


def test_confusion_matrix_is_square_over_the_scale() -> None:
    result = axis_agreement("g", [1, 5], [5, 1], 1, 5)
    assert len(result.confusion) == 5
    assert all(len(row) == 5 for row in result.confusion)
    assert result.confusion[0][4] == 1


def test_worst_disagreements_are_ordered_by_gap_then_stably() -> None:
    items = [
        Disagreement("b", "g", 5, 4, 1),
        Disagreement("a", "g", 5, 1, 4),
        Disagreement("a", "r", 5, 2, 3),
    ]
    assert [item.example_id for item in worst_disagreements(items, 2)] == ["a", "a"]
    assert worst_disagreements(items, 10) == worst_disagreements(list(reversed(items)), 10)


def test_length_bias_detects_a_monotone_relationship() -> None:
    result = length_bias("g", [10, 20, 30, 40, 50], [1, 2, 3, 4, 5])
    assert result.spearman == pytest.approx(1.0)


def test_length_bias_reports_why_it_could_not_correlate() -> None:
    result = length_bias("g", [10, 20, 30], [4, 4, 4])
    assert result.spearman is None
    assert result.note is not None


def test_position_consistency_counts_identical_scores() -> None:
    result = position_consistency("g", [5, 4, 3], [5, 4, 2])
    assert result.exact_agreement == pytest.approx(2 / 3)
    assert result.max_absolute_difference == 1


def test_position_consistency_of_an_empty_set_is_zero() -> None:
    assert position_consistency("g", [], []).n == 0


def test_self_preference_reports_the_gap_and_who_gained() -> None:
    result = self_preference("g", "judge-x", {"model-a": [5, 5, 4], "model-b": [3, 3, 2]})
    assert result.favoured == "model-a"
    assert result.gap > 1.0
