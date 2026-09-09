from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from evalgate.ingest.tokens import RegexTokenEstimator

ESTIMATOR = RegexTokenEstimator(name="t", pattern=r"\w+|[^\w\s]", tokens_per_word=1.3)


def test_spans_index_back_into_the_text() -> None:
    text = "Article 5 applies from 1 October 2023."
    assert [text[s:e] for s, e in ESTIMATOR.spans(text)] == ESTIMATOR.split(text)


def test_count_is_zero_for_whitespace() -> None:
    assert ESTIMATOR.count("   \n\t ") == 0


def test_count_scales_with_ratio() -> None:
    text = "one two three four five six seven eight nine ten"
    assert ESTIMATOR.count(text) == 13


@given(st.text(max_size=400))
def test_count_is_monotone_under_concatenation(text: str) -> None:
    assert ESTIMATOR.count(text + " zzz") >= ESTIMATOR.count(text)


@given(st.text(max_size=400))
def test_spans_are_ordered_and_disjoint(text: str) -> None:
    spans = ESTIMATOR.spans(text)
    for (_, end), (start, _) in pairwise(spans):
        assert end <= start


def test_fingerprint_captures_parameters() -> None:
    assert ESTIMATOR.fingerprint() == {
        "name": "t",
        "pattern": r"\w+|[^\w\s]",
        "tokens_per_word": 1.3,
    }


def test_different_ratio_gives_different_fingerprint() -> None:
    other = RegexTokenEstimator(name="t", pattern=r"\w+|[^\w\s]", tokens_per_word=1.0)
    assert other.fingerprint() != ESTIMATOR.fingerprint()


@pytest.mark.parametrize("text", ["", " ", "\n\n"])
def test_empty_text_has_no_spans(text: str) -> None:
    assert ESTIMATOR.spans(text) == []
