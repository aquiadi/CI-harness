"""Cross-encoder reranking.

Requires the ``ml`` extra. Scores every (query, chunk) pair jointly instead of
comparing independent embeddings, which is why it helps and why it costs an
order of magnitude more compute per query. The ablations measure whether that
trade is worth it on this corpus rather than assuming it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evalgate.embeddings.local import BackendUnavailableError
from evalgate.retrieval.base import Retrieved

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from sentence_transformers import CrossEncoder


@dataclass(slots=True)
class CrossEncoderReranker:
    """bge-reranker scoring of (query, chunk) pairs."""

    name: str
    model: str
    top_n: int
    batch_size: int
    device: str
    _encoder: CrossEncoder | None = field(default=None, repr=False, compare=False)

    def _load(self) -> CrossEncoder:
        if self._encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise BackendUnavailableError(
                    "reranking needs the ml extra: run `make ml`, or use retriever=hybrid"
                ) from exc
            self._encoder = CrossEncoder(self.model, device=self.device)
        return self._encoder

    def rerank(self, query: str, candidates: list[Retrieved], top_n: int) -> list[Retrieved]:
        """Rescore the head of the candidate list; leave the tail in place.

        Only the first ``top_n`` candidates are rescored -- reranking the whole
        pool costs linearly more for candidates that were never going to make
        the cut. The untouched tail keeps its fused order behind them.
        """
        head = candidates[: max(0, top_n)]
        tail = candidates[max(0, top_n) :]
        if not head:
            return list(candidates)

        scores = self._load().predict(
            [(query, result.text) for result in head],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        rescored = [
            Retrieved(
                chunk_id=result.chunk_id,
                score=float(score),
                rank=result.rank,
                text=result.text,
                doc_id=result.doc_id,
                doc_title=result.doc_title,
                section=result.section,
            )
            for result, score in zip(head, scores, strict=True)
        ]
        rescored.sort(key=lambda result: (-result.score, result.chunk_id))
        return rescored + tail

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this reranker and its parameters."""
        return {"name": self.name, "model": self.model, "top_n": self.top_n}
