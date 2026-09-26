"""Per-question movement and the bootstrap interval around the composite delta."""

from __future__ import annotations

from typing import Any

import pandas as pd

from evalgate.gate.items import (
    ANSWER_ITEMS,
    RETRIEVAL_ITEMS,
    bootstrap_interval,
    compare,
    compare_items,
    item_records,
)

AXES = ["groundedness", "relevance", "citation_correctness"]
WEIGHTS = {"groundedness": 0.3, "relevance": 0.2, "citation_correctness": 0.3, "recall_at_k": 0.2}


def rows(hits: list[float], grounded: list[int], answer: str = "a") -> pd.DataFrame:
    records: list[dict[str, Any]] = [
        {"task": "retrieval", "example_id": f"r-{i}", "recall_at_k": hit, "first_hit_rank": 0}
        for i, hit in enumerate(hits)
    ]
    records += [
        {
            "task": "answer",
            "example_id": f"a-{i}",
            "answer_hash": f"{answer}-{i}",
            "judge_groundedness": score,
            "judge_relevance": 5,
            "judge_citation_correctness": 5,
        }
        for i, score in enumerate(grounded)
    ]
    return pd.DataFrame(records)


def test_records_carry_one_entry_per_question() -> None:
    records = item_records(rows([1.0, 0.0], [5, 4]), AXES)
    assert [item["example_id"] for item in records[RETRIEVAL_ITEMS]] == ["r-0", "r-1"]
    assert records[ANSWER_ITEMS][1]["judge_groundedness"] == 4


def test_a_hit_that_became_a_miss_is_listed_as_worse() -> None:
    before = item_records(rows([1.0, 1.0], [5, 5]), AXES)
    after = item_records(rows([1.0, 0.0], [5, 3]), AXES)
    worse, better = compare_items(before, after, AXES)
    assert [(change.example_id, change.measure) for change in worse] == [
        ("a-1", "groundedness"),
        ("r-1", "recall_at_k"),
    ], "biggest drop first"
    assert not better


def test_a_rewritten_answer_is_flagged() -> None:
    before = item_records(rows([1.0], [5]), AXES)
    after = item_records(rows([1.0], [4], answer="b"), AXES)
    worse, _ = compare_items(before, after, AXES)
    assert worse[0].answer_changed


def test_an_unchanged_run_has_a_zero_delta_and_a_zero_width_interval() -> None:
    items = item_records(rows([1.0, 0.0, 1.0], [5, 4, 3]), AXES)
    interval = bootstrap_interval(items, items, WEIGHTS, (1, 5), 500, 0.95, 0)
    assert interval is not None
    assert interval.delta == 0.0
    assert interval.low == interval.high == 0.0


def test_the_interval_contains_the_delta_and_is_reproducible() -> None:
    before = item_records(rows([1.0] * 10, [5] * 10), AXES)
    after = item_records(rows([1.0] * 8 + [0.0] * 2, [5] * 9 + [1]), AXES)
    first = bootstrap_interval(before, after, WEIGHTS, (1, 5), 1000, 0.95, 7)
    second = bootstrap_interval(before, after, WEIGHTS, (1, 5), 1000, 0.95, 7)
    assert first == second, "seeded: the gate report must not change between identical runs"
    assert first is not None
    assert first.low < first.delta < first.high <= 0.0, "a drop, never an improvement"
    assert (first.n_retrieval, first.n_answers) == (10, 10)


def test_a_baseline_without_records_says_so_rather_than_inventing_a_comparison() -> None:
    comparison = compare({}, rows([1.0], [5]), AXES, WEIGHTS, (1, 5), 100, 0.95, 0)
    assert comparison.unavailable is not None
    assert comparison.interval is None and not comparison.worse
