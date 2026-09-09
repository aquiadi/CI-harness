"""Dense retrieval: exact cosine search over the embedded chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalgate.embeddings.base import Embedder
from evalgate.retrieval.base import Retrieved, to_results
from evalgate.retrieval.index import ChunkIndex


@dataclass(slots=True)
class DenseRetriever:
    """Nearest neighbours of the query embedding."""

    name: str
    k: int
    index: ChunkIndex
    embedder: Embedder

    def retrieve(self, query: str, k: int | None = None) -> list[Retrieved]:
        """Return the k nearest chunks, best first."""
        limit = self.k if k is None else k
        scored = self.index.dense_search(self.embedder.embed_query(query), limit)
        return to_results(scored, self.index.chunks)

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this retriever and its parameters."""
        return {"name": self.name, "k": self.k, "embedder": self.embedder.fingerprint()}
