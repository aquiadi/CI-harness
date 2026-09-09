"""Sparse retrieval: BM25 over the same chunk set.

Kept as a first-class retriever rather than as a hybrid component only, because
it is the honest baseline: regulatory questions often share exact vocabulary
with the instrument ("authorised CBAM declarant", "Annex I goods"), and a dense
retriever that cannot beat BM25 on those is not earning its embedding cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalgate.retrieval.base import Retrieved, to_results
from evalgate.retrieval.index import ChunkIndex


@dataclass(slots=True)
class BM25Retriever:
    """Lexical retrieval with the index's BM25 parameters."""

    name: str
    k: int
    index: ChunkIndex

    def retrieve(self, query: str, k: int | None = None) -> list[Retrieved]:
        """Return the k highest scoring chunks, best first."""
        limit = self.k if k is None else k
        return to_results(self.index.sparse_search(query, limit), self.index.chunks)

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this retriever and its parameters."""
        return {"name": self.name, "k": self.k, "sparse": self.index.meta.sparse.model_dump()}
