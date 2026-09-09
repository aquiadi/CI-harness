"""Retrieval and aggregate metrics.

Definitions are spelled out because a metric whose definition is implicit is a
metric nobody can check.

*recall@k*: the share of eval questions for which at least one of the top k
retrieved chunks contains a gold span. Binary per question.

*MRR*: mean of 1/(rank of the first hit), one-based, zero when there is no hit
in the top k.

*nDCG@10*: computed under a single-relevant-item assumption -- the ideal
ranking puts the gold evidence first, so IDCG is 1 and nDCG reduces to
1/log2(rank + 1). Gold evidence often appears in more than one chunk (spans
straddle boundaries, chunkers overlap), and counting every such chunk as
relevant would make the ideal ranking depend on the chunker, which is exactly
what the ablation varies. With several gold spans, each is scored separately
and the results averaged.

*p50/p95 latency*: linear-interpolated percentiles over per-question wall-clock
seconds. For replayed model calls this is the latency measured when the call
was recorded, not the time taken to read the cassette; a run made in replay
mode is marked as such and the report says so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def recall_at_k(hit_rank: int | None, k: int) -> float:
    """1.0 when a gold span was found within the top k."""
    return 1.0 if hit_rank is not None and hit_rank < k else 0.0


def reciprocal_rank(hit_rank: int | None) -> float:
    """1/(one-based rank of the first hit), else 0."""
    return 0.0 if hit_rank is None else 1.0 / (hit_rank + 1)


def ndcg_at_k(hit_rank: int | None, k: int) -> float:
    """Discounted gain of the first hit under a single-relevant-item ideal."""
    if hit_rank is None or hit_rank >= k:
        return 0.0
    return 1.0 / math.log2(hit_rank + 2)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; 0.0 for an empty sample."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; 0.0 for an empty sample."""
    return float(np.mean(values)) if len(values) else 0.0


def normalise_score(score: float, scale_min: int, scale_max: int) -> float:
    """Map an integer rubric score onto 0..1 so axes can be combined."""
    span = scale_max - scale_min
    if span <= 0:
        raise ValueError("judge scale must increase")
    return (score - scale_min) / span


def composite_quality(components: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean of already-normalised components.

    Refuses to run on weights that do not sum to one, and on a component the
    weights do not mention: a composite whose parts silently changed is not
    comparable to the baseline it is being checked against.
    """
    if not weights:
        raise ValueError("composite quality needs weights")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"composite weights must sum to 1.0, got {total}")
    missing = sorted(set(weights) - set(components))
    if missing:
        raise ValueError(f"no measurement for weighted component(s): {', '.join(missing)}")
    return sum(components[name] * weight for name, weight in weights.items())
