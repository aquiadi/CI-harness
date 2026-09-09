"""What a judge is given and what it must return."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from evalgate.evalsets.schemas import JudgeScore
from evalgate.retrieval.base import Retrieved


@dataclass(frozen=True, slots=True)
class JudgeInput:
    """One answer to be graded, with everything it was grounded in."""

    example_id: str
    question: str
    answer: str
    context: list[Retrieved]
    gold_spans: list[str] = field(default_factory=list)
    # Set for the position-bias probe, which grades the same answer twice with
    # the context in a different order.
    variant: str = "primary"
    # The generator that produced this answer, for the self-preference probe.
    generator: str = "reference"


@runtime_checkable
class Judge(Protocol):
    """Grades one answer on the configured axes."""

    name: str
    model: str

    def score(self, item: JudgeInput) -> JudgeScore:
        """Return scores, or raise. Never return a default."""
        ...

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this judge and its parameters, for the run record."""
        ...
