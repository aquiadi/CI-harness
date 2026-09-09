"""The retrieval interface, and the ordering rule every retriever obeys.

Ties are broken by chunk id, not by whatever order the backend happened to
produce. BM25 in particular returns zero scores for out-of-vocabulary queries,
and an index-order tie-break would make results depend on ingestion order --
one of those defects that only shows up as unreproducible eval numbers weeks
later.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from evalgate.chunking.base import Chunk


@dataclass(frozen=True, slots=True)
class Retrieved:
    """One retrieved chunk, with its score and position."""

    chunk_id: str
    score: float
    rank: int
    text: str
    doc_id: str
    doc_title: str
    section: str | None = None


@runtime_checkable
class Retriever(Protocol):
    """Returns the k chunks most relevant to a query."""

    name: str
    k: int

    def retrieve(self, query: str, k: int | None = None) -> list[Retrieved]:
        """Return at most k results, best first."""
        ...

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this retriever and its parameters, for the run record."""
        ...


def rank_scores(scores: Iterable[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """Sort (chunk_id, score) pairs deterministically and take the top k."""
    ordered = sorted(scores, key=lambda pair: (-pair[1], pair[0]))
    return ordered[: max(0, k)]


def to_results(
    scored: list[tuple[str, float]],
    chunks: Mapping[str, Chunk],
) -> list[Retrieved]:
    """Attach chunk content to scored ids, preserving order."""
    results: list[Retrieved] = []
    for rank, (chunk_id, score) in enumerate(scored):
        chunk = chunks[chunk_id]
        results.append(
            Retrieved(
                chunk_id=chunk_id,
                score=float(score),
                rank=rank,
                text=chunk.text,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                section=chunk.section,
            )
        )
    return results
