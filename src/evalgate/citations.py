"""Extracting citations from a generated answer.

This is a regex, and it is worth being explicit about why that does not
contradict the rule in docs/PRINCIPLES.md that judge output is never parsed with a
regex. The rule exists because a half-matching regex over a judge's prose would
manufacture a score that looks real. Here the input is an *answer*, the output
is a set of chunk ids whose validity is then checked against the context, and a
missed or spurious match shows up as a citation the context does not contain --
which is measured, not silently accepted. Judge scores still arrive only
through a validated tool call.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Chunk ids are `{doc_id}#{ordinal}`; doc ids are filename stems.
CITATION = re.compile(r"\[([A-Za-z0-9_.\-]+#\d+)\]")


@dataclass(frozen=True, slots=True)
class CitationAudit:
    """What an answer cited, and whether those citations exist in its context."""

    cited: list[str]
    valid: list[str]
    dangling: list[str]

    @property
    def count(self) -> int:
        """How many citations the answer made."""
        return len(self.cited)

    @property
    def precision(self) -> float:
        """Share of citations that name a chunk actually in the context."""
        return len(self.valid) / self.count if self.count else 0.0


def extract_citations(answer: str) -> list[str]:
    """Every bracketed chunk id in the answer, in order, without duplicates."""
    return list(dict.fromkeys(CITATION.findall(answer)))


def audit_citations(answer: str, context_chunk_ids: Sequence[str]) -> CitationAudit:
    """Split an answer's citations into those the context contains and those it does not."""
    available = set(context_chunk_ids)
    cited = extract_citations(answer)
    valid = [chunk_id for chunk_id in cited if chunk_id in available]
    dangling = [chunk_id for chunk_id in cited if chunk_id not in available]
    return CitationAudit(cited=cited, valid=valid, dangling=dangling)
