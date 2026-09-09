"""The embedding interface every backend implements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vectors = NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Documents and queries are embedded through separate methods because the
    default model (bge-small) is trained asymmetrically: queries take an
    instruction prefix that documents must not get. A single ``embed`` method
    would silently apply the wrong one.
    """

    name: str
    model: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Embed passages for indexing. Shape (len(texts), dim)."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """Embed one query. Shape (dim,)."""
        ...

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this backend and its parameters, for the run record."""
        ...


def as_float32(array: NDArray[Any]) -> Vectors:
    """Narrow a numpy result to the vector dtype the pipeline stores."""
    return cast(Vectors, array.astype(np.float32, copy=False))


def l2_normalise(vectors: Vectors) -> Vectors:
    """Scale rows to unit length, leaving all-zero rows alone."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return as_float32(vectors / norms)
