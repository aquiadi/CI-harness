"""Judge bias probes.

Each probe asks whether a score moves for a reason that has nothing to do with
answer quality. None of them proves a judge is unbiased; each can show that one
is.

*Position swap*: grade the same answer twice, with the retrieved context in the
opposite order. The answer is unchanged, so any score change is the judge
responding to presentation.

*Length bias*: correlate answer length with score across the eval set. A strong
positive correlation means longer answers score better, which is the failure
mode that quietly rewards padding.

*Self-preference*: run two generators over the same questions and let one judge
grade both. A judge that systematically prefers its own model family is not
measuring answer quality, and any leaderboard built on it is measuring
lineage.

A correlation here is evidence, not proof: longer answers may genuinely be
better, and one model family may genuinely answer better. The report says the
size of the effect and leaves the interpretation visible rather than declaring
a verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class LengthBias:
    """Correlation between answer length and score, per axis."""

    axis: str
    n: int
    spearman: float | None
    p_value: float | None
    mean_tokens: float
    note: str | None = None


@dataclass(frozen=True, slots=True)
class PositionConsistency:
    """Stability of a judge's score when the context order is reversed."""

    axis: str
    n: int
    exact_agreement: float
    mean_absolute_difference: float
    max_absolute_difference: int


@dataclass(frozen=True, slots=True)
class SelfPreference:
    """Mean score a judge gives each generator, on one axis."""

    axis: str
    judge_model: str
    n: int
    means: dict[str, float]
    gap: float
    favoured: str | None


def length_bias(axis: str, lengths: Sequence[int], scores: Sequence[int]) -> LengthBias:
    """Spearman correlation between answer length and score."""
    count = len(scores)
    if count != len(lengths):
        raise ValueError("lengths and scores must align")
    if count < 3 or len(set(scores)) == 1 or len(set(lengths)) == 1:
        return LengthBias(
            axis=axis,
            n=count,
            spearman=None,
            p_value=None,
            mean_tokens=float(np.mean(lengths)) if lengths else 0.0,
            note="too few items, or no variance to correlate",
        )
    result = stats.spearmanr(lengths, scores)
    return LengthBias(
        axis=axis,
        n=count,
        spearman=float(result.statistic),
        p_value=float(result.pvalue),
        mean_tokens=float(np.mean(lengths)),
    )


def position_consistency(
    axis: str, primary: Sequence[int], swapped: Sequence[int]
) -> PositionConsistency:
    """How often the score survives reversing the context order."""
    if len(primary) != len(swapped):
        raise ValueError("primary and swapped score lists must align")
    if not primary:
        return PositionConsistency(axis, 0, 0.0, 0.0, 0)
    difference = np.abs(np.asarray(primary, dtype=int) - np.asarray(swapped, dtype=int))
    return PositionConsistency(
        axis=axis,
        n=len(primary),
        exact_agreement=float(np.mean(difference == 0)),
        mean_absolute_difference=float(np.mean(difference)),
        max_absolute_difference=int(np.max(difference)),
    )


def self_preference(
    axis: str, judge_model: str, scores_by_generator: Mapping[str, Sequence[int]]
) -> SelfPreference:
    """Mean score per generator, and the gap between the best and worst."""
    means = {
        generator: float(np.mean(scores)) if len(scores) else 0.0
        for generator, scores in scores_by_generator.items()
    }
    counts = {len(scores) for scores in scores_by_generator.values()}
    gap = max(means.values()) - min(means.values()) if means else 0.0
    favoured = max(means, key=lambda key: means[key]) if means else None
    return SelfPreference(
        axis=axis,
        judge_model=judge_model,
        n=min(counts) if counts else 0,
        means=means,
        gap=gap,
        favoured=favoured,
    )
