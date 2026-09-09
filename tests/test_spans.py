from __future__ import annotations

from evalgate.evaluation.spans import (
    contains_any,
    contains_span,
    first_hit_rank,
    flatten,
    hit_ranks,
)


def test_line_wrapping_does_not_break_a_match() -> None:
    chunk = "shall take a decision within 120\ncalendar days of receipt"
    assert contains_span(chunk, "within 120 calendar days")


def test_matching_is_case_sensitive() -> None:
    """Case carries meaning in regulatory text: Article is not article."""
    assert not contains_span("see article 5", "Article 5")


def test_absent_span_does_not_match() -> None:
    assert not contains_span("the quarterly report", "the annual report")


def test_flatten_collapses_all_whitespace() -> None:
    assert flatten(" a \n\t b  ") == "a b"


def test_contains_any_needs_only_one() -> None:
    assert contains_any("alpha beta", ["gamma", "beta"])


def test_first_hit_rank_reports_position() -> None:
    assert first_hit_rank(["nope", "gold here", "gold here"], ["gold here"]) == 1


def test_first_hit_rank_is_none_when_absent() -> None:
    assert first_hit_rank(["a", "b"], ["c"]) is None


def test_hit_ranks_lists_every_hit() -> None:
    assert hit_ranks(["gold", "no", "gold"], ["gold"]) == [0, 2]
