"""What changed question by question, and how much of it a sample this size can show.

The gate's thresholds are applied to averages over fifteen questions, where one
question moves recall by nearly seven points. An average that dropped says
nothing about which question got worse, and a reviewer deciding whether a
regression is real needs exactly that. So the baseline keeps one record per
question, and a gate run lists the questions that went from a hit to a miss or
lost a judge point.

The interval is the other half. A paired bootstrap resamples the questions and
recomputes the composite delta on each resample, which says how far the delta
could have moved under a different sample of questions of the same size. It is
reported, never gated: the offline stack is deterministic, so the delta itself
is exact, and a gate that passed a drop because the interval was wide would be
widening its own threshold -- which D-0031's spirit and the constitution both
forbid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TASK_RETRIEVAL = "retrieval"
TASK_ANSWER = "answer"
RETRIEVAL_ITEMS = "retrieval"
ANSWER_ITEMS = "answers"
RECALL = "recall_at_k"
JUDGE_PREFIX = "judge_"


@dataclass(frozen=True, slots=True)
class ItemChange:
    """One question whose score moved."""

    example_id: str
    measure: str
    before: float
    after: float
    answer_changed: bool = False


@dataclass(frozen=True, slots=True)
class Interval:
    """A bootstrap interval around the composite delta."""

    delta: float
    low: float
    high: float
    confidence: float
    resamples: int
    n_retrieval: int
    n_answers: int


@dataclass(frozen=True, slots=True)
class ItemComparison:
    """Per-question movement between the baseline and a run."""

    worse: list[ItemChange]
    better: list[ItemChange]
    interval: Interval | None
    unavailable: str | None = None


def _value(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if np.isnan(number) else number


def item_records(rows: pd.DataFrame, axes: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """One small record per question, in the shape the baseline stores."""
    records: dict[str, list[dict[str, Any]]] = {RETRIEVAL_ITEMS: [], ANSWER_ITEMS: []}
    if rows.empty:
        return records
    for row in rows.to_dict("records"):
        if row.get("task") == TASK_RETRIEVAL:
            records[RETRIEVAL_ITEMS].append(
                {
                    "example_id": str(row["example_id"]),
                    RECALL: _value(row.get(RECALL)),
                    "first_hit_rank": int(row.get("first_hit_rank", -1)),
                }
            )
        elif row.get("task") == TASK_ANSWER:
            record: dict[str, Any] = {
                "example_id": str(row["example_id"]),
                "answer_hash": str(row.get("answer_hash", "")),
            }
            for axis in axes:
                record[f"{JUDGE_PREFIX}{axis}"] = _value(row.get(f"{JUDGE_PREFIX}{axis}"))
            records[ANSWER_ITEMS].append(record)
    for kind in records:
        records[kind].sort(key=lambda item: str(item["example_id"]))
    return records


def _by_id(items: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(item["example_id"]): item for item in items}


def compare_items(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    current: Mapping[str, Sequence[Mapping[str, Any]]],
    axes: Sequence[str],
) -> tuple[list[ItemChange], list[ItemChange]]:
    """Questions that got worse and questions that got better, worst first."""
    worse: list[ItemChange] = []
    better: list[ItemChange] = []

    before = _by_id(baseline.get(RETRIEVAL_ITEMS, []))
    for example_id, item in sorted(_by_id(current.get(RETRIEVAL_ITEMS, [])).items()):
        old = before.get(example_id)
        if old is None:
            continue
        was, now = _value(old.get(RECALL)), _value(item.get(RECALL))
        if was is None or now is None or was == now:
            continue
        change = ItemChange(example_id, RECALL, was, now)
        (worse if now < was else better).append(change)

    before = _by_id(baseline.get(ANSWER_ITEMS, []))
    for example_id, item in sorted(_by_id(current.get(ANSWER_ITEMS, [])).items()):
        old = before.get(example_id)
        if old is None:
            continue
        changed = old.get("answer_hash") != item.get("answer_hash")
        for axis in axes:
            key = f"{JUDGE_PREFIX}{axis}"
            was, now = _value(old.get(key)), _value(item.get(key))
            if was is None or now is None or was == now:
                continue
            change = ItemChange(example_id, axis, was, now, answer_changed=changed)
            (worse if now < was else better).append(change)

    worse.sort(key=lambda change: (change.after - change.before, change.example_id))
    better.sort(key=lambda change: (change.before - change.after, change.example_id))
    return worse, better


def _paired(
    baseline: Sequence[Mapping[str, Any]], current: Sequence[Mapping[str, Any]], key: str
) -> tuple[np.ndarray, np.ndarray]:
    before, after = _by_id(baseline), _by_id(current)
    shared = sorted(
        example_id
        for example_id in set(before) & set(after)
        if _value(before[example_id].get(key)) is not None
        and _value(after[example_id].get(key)) is not None
    )
    return (
        np.array([float(before[example_id][key]) for example_id in shared]),
        np.array([float(after[example_id][key]) for example_id in shared]),
    )


def bootstrap_interval(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    current: Mapping[str, Sequence[Mapping[str, Any]]],
    weights: Mapping[str, float],
    scale: tuple[int, int],
    resamples: int,
    confidence: float,
    seed: int,
) -> Interval | None:
    """A paired bootstrap interval for the composite quality delta.

    Retrieval and answer questions are resampled independently, because the
    composite averages them independently. None when a weighted component has
    no question measured on both sides.
    """
    low_score, high_score = scale
    span = float(high_score - low_score)
    columns: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    for component, weight in sorted(weights.items()):
        if component == RECALL:
            kind = RETRIEVAL_ITEMS
            before, after = _paired(
                baseline.get(RETRIEVAL_ITEMS, []), current.get(RETRIEVAL_ITEMS, []), RECALL
            )
        else:
            kind = ANSWER_ITEMS
            before, after = _paired(
                baseline.get(ANSWER_ITEMS, []),
                current.get(ANSWER_ITEMS, []),
                f"{JUDGE_PREFIX}{component}",
            )
            before, after = (before - low_score) / span, (after - low_score) / span
        if len(before) == 0:
            return None
        columns.append((kind, float(weight), before, after))

    rng = np.random.default_rng(seed)
    deltas = np.zeros(resamples)
    point = 0.0
    # One draw per kind of question: every axis of an answer moves with that
    # answer, and the retrieval questions are resampled on their own.
    draws: dict[tuple[str, int], np.ndarray] = {}
    sizes: dict[str, int] = {}
    for kind, weight, before, after in columns:
        size = len(before)
        sizes[kind] = max(sizes.get(kind, 0), size)
        if (kind, size) not in draws:
            draws[(kind, size)] = rng.integers(0, size, size=(resamples, size))
        index = draws[(kind, size)]
        deltas += weight * (after[index].mean(axis=1) - before[index].mean(axis=1))
        point += weight * float(after.mean() - before.mean())
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(deltas, [tail, 1.0 - tail])
    return Interval(
        delta=point,
        low=float(low),
        high=float(high),
        confidence=confidence,
        resamples=resamples,
        n_retrieval=sizes.get(RETRIEVAL_ITEMS, 0),
        n_answers=sizes.get(ANSWER_ITEMS, 0),
    )


def compare(
    baseline_items: Mapping[str, Sequence[Mapping[str, Any]]],
    rows: pd.DataFrame,
    axes: Sequence[str],
    weights: Mapping[str, float],
    scale: tuple[int, int],
    resamples: int,
    confidence: float,
    seed: int,
) -> ItemComparison:
    """Everything the gate report says about individual questions."""
    if not baseline_items:
        return ItemComparison(
            worse=[],
            better=[],
            interval=None,
            unavailable=(
                "the baseline predates per-question records; re-freeze it to list which "
                "questions moved"
            ),
        )
    current = item_records(rows, axes)
    worse, better = compare_items(baseline_items, current, axes)
    interval = bootstrap_interval(
        baseline_items, current, weights, scale, resamples, confidence, seed
    )
    return ItemComparison(worse=worse, better=better, interval=interval)
