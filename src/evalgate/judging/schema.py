"""The judgment schema, built from config.

The axes and the scale live in ``configs/judge/``, so the pydantic model that
validates a judgment is constructed at runtime from those values rather than
written out with a hardcoded 1-5 bound. Adding a fourth axis is a config
change; the tool schema, the validation bounds and the report columns all
follow from it.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, create_model

RATIONALE_SUFFIX = "_rationale"
SCORE_DESCRIPTION = "Integer score for {axis}, from {low} (worst) to {high} (best)."
RATIONALE_DESCRIPTION = (
    "One or two sentences justifying the {axis} score, naming the specific chunk id "
    "or quoted span that drove it."
)


def build_judgment_model(axes: Sequence[str], low: int, high: int) -> type[BaseModel]:
    """A pydantic model with one bounded score and one rationale per axis."""
    if not axes:
        raise ValueError("a judge needs at least one axis")
    if high <= low:
        raise ValueError(f"judge scale must increase: got {low}..{high}")

    fields: dict[str, object] = {}
    for axis in axes:
        fields[axis] = (
            int,
            Field(
                ge=low, le=high, description=SCORE_DESCRIPTION.format(axis=axis, low=low, high=high)
            ),
        )
        fields[f"{axis}{RATIONALE_SUFFIX}"] = (
            str,
            Field(min_length=1, description=RATIONALE_DESCRIPTION.format(axis=axis)),
        )
    return create_model(  # type: ignore[call-overload, no-any-return]
        "Judgment",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def split_judgment(
    payload: dict[str, object], axes: Sequence[str]
) -> tuple[dict[str, int], dict[str, str]]:
    """Separate a validated judgment into scores and rationales."""
    scores = {axis: int(payload[axis]) for axis in axes}  # type: ignore[call-overload]
    rationales = {axis: str(payload.get(f"{axis}{RATIONALE_SUFFIX}", "")) for axis in axes}
    return scores, rationales
