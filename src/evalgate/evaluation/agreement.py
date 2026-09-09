"""Agreement between a judge and human labels.

Cohen's kappa and its quadratic-weighted variant, per axis, plus the confusion
matrix and the disagreements worth reading. Kappa rather than raw agreement
because these scales are skewed: if 80 per cent of answers are a 5, a judge that
always says 5 agrees 80 per cent of the time and has learned nothing. Quadratic
weighting alongside it because on an ordinal scale a 5-versus-4 disagreement is
not the same failure as 5-versus-1.

Kappa is undefined when either rater gives the same score to everything -- the
expected-agreement term goes to one and the statistic divides by zero. That is
reported as undefined, with the reason, rather than as 0.0. A 0.0 says "no
better than chance"; undefined says "this data cannot answer the question", and
conflating them is exactly the kind of quiet lie this repo exists to avoid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

UNDEFINED_REASON_CONSTANT = "one rater gave the same score to every item"
UNDEFINED_REASON_EMPTY = "no overlapping labelled items"


@dataclass(frozen=True, slots=True)
class AxisAgreement:
    """How well a judge matched human labels on one axis."""

    axis: str
    n: int
    kappa: float | None
    quadratic_kappa: float | None
    undefined_reason: str | None
    exact_agreement: float
    within_one: float
    mean_human: float
    mean_judge: float
    mean_signed_error: float
    mean_absolute_error: float
    confusion: list[list[int]] = field(default_factory=list)

    @property
    def is_defined(self) -> bool:
        """Whether kappa could be computed at all."""
        return self.kappa is not None


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One item where judge and human differed, worth a human's attention."""

    example_id: str
    axis: str
    human: int
    judge: int
    gap: int
    rationale: str = ""
    answer: str = ""


def _kappa(human: np.ndarray, judge: np.ndarray, labels: list[int], weights: str | None) -> float:
    return float(cohen_kappa_score(human, judge, labels=labels, weights=weights))


def axis_agreement(
    axis: str,
    human_scores: Sequence[int],
    judge_scores: Sequence[int],
    scale_min: int,
    scale_max: int,
) -> AxisAgreement:
    """Compute agreement statistics for one axis."""
    if len(human_scores) != len(judge_scores):
        raise ValueError("human and judge score lists must align")

    labels = list(range(scale_min, scale_max + 1))
    human = np.asarray(human_scores, dtype=int)
    judge = np.asarray(judge_scores, dtype=int)
    count = int(human.size)

    if count == 0:
        return AxisAgreement(
            axis=axis,
            n=0,
            kappa=None,
            quadratic_kappa=None,
            undefined_reason=UNDEFINED_REASON_EMPTY,
            exact_agreement=0.0,
            within_one=0.0,
            mean_human=0.0,
            mean_judge=0.0,
            mean_signed_error=0.0,
            mean_absolute_error=0.0,
            confusion=[[0] * len(labels) for _ in labels],
        )

    difference = judge - human
    matrix = confusion_matrix(human, judge, labels=labels).tolist()
    constant = len(set(human.tolist())) == 1 or len(set(judge.tolist())) == 1

    return AxisAgreement(
        axis=axis,
        n=count,
        kappa=None if constant else _kappa(human, judge, labels, None),
        quadratic_kappa=None if constant else _kappa(human, judge, labels, "quadratic"),
        undefined_reason=UNDEFINED_REASON_CONSTANT if constant else None,
        exact_agreement=float(np.mean(difference == 0)),
        within_one=float(np.mean(np.abs(difference) <= 1)),
        mean_human=float(np.mean(human)),
        mean_judge=float(np.mean(judge)),
        mean_signed_error=float(np.mean(difference)),
        mean_absolute_error=float(np.mean(np.abs(difference))),
        confusion=matrix,
    )


def worst_disagreements(disagreements: Sequence[Disagreement], limit: int) -> list[Disagreement]:
    """The largest gaps, biggest first, ties broken by example id for stability."""
    return sorted(disagreements, key=lambda item: (-item.gap, item.example_id, item.axis))[:limit]
