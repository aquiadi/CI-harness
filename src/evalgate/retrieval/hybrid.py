"""Hybrid retrieval by reciprocal rank fusion.

RRF rather than a weighted sum of scores: cosine similarity and BM25 are not on
a comparable scale, and normalising them introduces a tuning knob whose value
would have to be defended for every corpus. RRF uses only the ranks, so it has
one parameter with a well-understood effect and cannot be gamed by one
retriever's score distribution.

Optionally reranks the fused candidate list with a cross-encoder, which is the
one place in retrieval where paying for a second model is usually worth it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from evalgate.embeddings.base import Embedder
from evalgate.retrieval.base import Retrieved, rank_scores, to_results
from evalgate.retrieval.index import ChunkIndex


@runtime_checkable
class Reranker(Protocol):
    """Reorders a candidate list against the query."""

    name: str
    model: str

    def rerank(self, query: str, candidates: list[Retrieved], top_n: int) -> list[Retrieved]:
        """Return candidates reordered, best first."""
        ...

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this reranker, for the run record."""
        ...


@dataclass(slots=True)
class HybridRRFRetriever:
    """Reciprocal rank fusion of the dense and sparse lists."""

    name: str
    k: int
    candidate_k: int
    rrf_k: int
    dense_weight: float
    sparse_weight: float
    index: ChunkIndex
    embedder: Embedder
    reranker: Reranker | None = None
    rerank_top_n: int = 0

    def retrieve(self, query: str, k: int | None = None) -> list[Retrieved]:
        """Fuse the two ranked lists, optionally rerank, and return the top k."""
        limit = self.k if k is None else k
        pool = max(self.candidate_k, limit)

        dense = self.index.dense_search(self.embedder.embed_query(query), pool)
        sparse = self.index.sparse_search(query, pool)

        fused: dict[str, float] = {}
        for weight, ranked in ((self.dense_weight, dense), (self.sparse_weight, sparse)):
            for rank, (chunk_id, _) in enumerate(ranked):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (self.rrf_k + rank + 1)

        if self.reranker is None:
            return to_results(rank_scores(fused.items(), limit), self.index.chunks)

        candidates = to_results(rank_scores(fused.items(), pool), self.index.chunks)
        top_n = self.rerank_top_n or pool
        reranked = self.reranker.rerank(query, candidates, top_n)
        return [
            Retrieved(
                chunk_id=result.chunk_id,
                score=result.score,
                rank=rank,
                text=result.text,
                doc_id=result.doc_id,
                doc_title=result.doc_title,
                section=result.section,
            )
            for rank, result in enumerate(reranked[:limit])
        ]

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this retriever and its parameters."""
        return {
            "name": self.name,
            "k": self.k,
            "candidate_k": self.candidate_k,
            "rrf_k": self.rrf_k,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "embedder": self.embedder.fingerprint(),
            "reranker": self.reranker.fingerprint() if self.reranker else None,
            "rerank_top_n": self.rerank_top_n if self.reranker else None,
        }
