"""Local sentence-transformers embeddings: the default backend.

Requires the ``ml`` extra (``make ml``). Kept out of the default dependency set
so that lint, types, tests and replay-mode CI do not install torch; see
docs/DECISIONS.md D-0004.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from evalgate.embeddings.base import Vectors, as_float32, l2_normalise
from evalgate.errors import EvalgateError

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from sentence_transformers import SentenceTransformer


class BackendUnavailableError(EvalgateError):
    """Raised when a backend's dependencies or credentials are missing."""


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """bge-family encoder run locally on CPU."""

    name: str
    model: str
    dim: int
    batch_size: int
    normalize: bool
    query_prefix: str
    document_prefix: str
    device: str
    _encoder: SentenceTransformer | None = field(default=None, repr=False, compare=False)

    def _load(self) -> SentenceTransformer:
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise BackendUnavailableError(
                    "embedder=local needs the ml extra: run `make ml`, "
                    "or use embedder=hashed for a dependency-free run"
                ) from exc
            self._encoder = SentenceTransformer(self.model, device=self.device)
        return self._encoder

    def _encode(self, texts: Sequence[str]) -> Vectors:
        encoder = self._load()
        raw = encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        matrix = np.asarray(raw, dtype=np.float32).reshape(len(texts), -1)
        if matrix.shape[1] != self.dim:
            raise BackendUnavailableError(
                f"{self.model} produced dim {matrix.shape[1]}, config says {self.dim}"
            )
        return l2_normalise(matrix) if self.normalize else as_float32(matrix)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Embed passages for indexing."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> Vectors:
        """Embed one query, with the model's query instruction prefix."""
        return as_float32(self._encode([self.query_prefix + text])[0])

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this backend and its parameters."""
        return {
            "name": self.name,
            "model": self.model,
            "dim": self.dim,
            "normalize": self.normalize,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
        }
