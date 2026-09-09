"""What a generator is given and what it returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from evalgate.retrieval.base import Retrieved


@dataclass(frozen=True, slots=True)
class GenerationInput:
    """One question and the context retrieved for it."""

    example_id: str
    question: str
    context: list[Retrieved]


@dataclass(frozen=True, slots=True)
class Generated:
    """An answer and what producing it cost."""

    example_id: str
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    context_tokens: int
    replayed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def cost_usd(self, input_usd_per_mtok: float, output_usd_per_mtok: float) -> float:
        """Measured cost: zero for a generator that makes no API call."""
        return (
            self.input_tokens * input_usd_per_mtok + self.output_tokens * output_usd_per_mtok
        ) / 1_000_000.0


@runtime_checkable
class Generator(Protocol):
    """Produces an answer from retrieved context."""

    name: str
    model: str

    def generate(self, item: GenerationInput) -> Generated:
        """Answer one question."""
        ...

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this generator, for the run record."""
        ...
