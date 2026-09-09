"""Token estimation.

Chunk sizes are expressed in tokens, but the exact BPE tokenizer of the
generation model is not available offline, and calling the API to count tokens
for every chunk would cost money and make chunk boundaries depend on the
network. So boundaries are decided by a deterministic regex word count scaled
by a configured words-to-tokens ratio.

This is an estimate, and it is the right kind of estimate: stable across
machines and processes, monotone in text length, and wrong by a roughly
constant factor. What the ablations need is for "512 tokens" to mean the same
thing in every run, not to match Anthropic's tokenizer exactly. Billed tokens
are read from API usage and never estimated.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

Span = tuple[int, int]


@runtime_checkable
class TokenEstimator(Protocol):
    """Estimates the token length of a string and locates its atoms."""

    def spans(self, text: str) -> list[Span]:
        """Return (start, end) character offsets of each counted atom."""
        ...

    def count(self, text: str) -> int:
        """Return the estimated token count."""
        ...

    def fingerprint(self) -> dict[str, str | float]:
        """Identity of this estimator, for the run record."""
        ...


@dataclass(frozen=True, slots=True)
class RegexTokenEstimator:
    """Word-ish atoms from a regex, scaled to approximate BPE tokens."""

    name: str
    pattern: str
    tokens_per_word: float
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compile the pattern once; the dataclass is frozen so bypass setattr."""
        object.__setattr__(self, "_compiled", re.compile(self.pattern))

    def spans(self, text: str) -> list[Span]:
        """Character offsets of each word or standalone punctuation mark."""
        return [match.span() for match in self._compiled.finditer(text)]

    def split(self, text: str) -> list[str]:
        """The atoms themselves."""
        return [text[start:end] for start, end in self.spans(text)]

    def count(self, text: str) -> int:
        """Estimated token count, rounded up."""
        return math.ceil(len(self.spans(text)) * self.tokens_per_word)

    def fingerprint(self) -> dict[str, str | float]:
        """Identity of this estimator, for the run record."""
        return {
            "name": self.name,
            "pattern": self.pattern,
            "tokens_per_word": self.tokens_per_word,
        }
