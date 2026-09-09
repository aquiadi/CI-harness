"""A dependency-free deterministic embedder.

A signed random projection of hashed token n-grams: for each n-gram, a BLAKE2b
digest picks a coordinate and a sign, and the counts are accumulated and
L2-normalised. This is the hashing trick with a random sign, which preserves
inner products in expectation -- weak next to a trained encoder, but real
vectors doing real arithmetic.

It exists because three things need embeddings that cost nothing and never
vary: the test suite, replay-mode CI, and anyone reproducing this repo from
behind a network policy that blocks model hosts. Because it is a real backend
rather than a test double, those three exercise the same retrieval code as a
production run. Python's ``hash()`` is deliberately not used: it is salted per
process, which would make the index depend on which process built it.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evalgate.embeddings.base import Vectors, as_float32, l2_normalise

_WORD = re.compile(r"\w+")
_DIGEST_BYTES = 8


@dataclass(slots=True)
class HashEmbedder:
    """Signed random projection of hashed word n-grams."""

    name: str
    model: str
    dim: int
    batch_size: int
    normalize: bool
    query_prefix: str
    document_prefix: str
    ngram_range: tuple[int, int] | list[int]
    seed: int

    def _grams(self, text: str) -> list[str]:
        words = _WORD.findall(text.lower())
        low, high = int(self.ngram_range[0]), int(self.ngram_range[1])
        grams: list[str] = []
        for size in range(max(1, low), max(low, high) + 1):
            grams.extend(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))
        return grams

    def _embed_one(self, text: str) -> Vectors:
        vector = np.zeros(self.dim, dtype=np.float32)
        for gram in self._grams(text):
            payload = f"{self.seed}\x1f{gram}".encode()
            digest = hashlib.blake2b(payload, digest_size=_DIGEST_BYTES).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        return vector

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Embed passages for indexing."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.stack([self._embed_one(self.document_prefix + text) for text in texts])
        return l2_normalise(matrix) if self.normalize else as_float32(matrix)

    def embed_query(self, text: str) -> Vectors:
        """Embed one query."""
        vector = self._embed_one(self.query_prefix + text)
        if self.normalize:
            return as_float32(l2_normalise(vector.reshape(1, -1))[0])
        return as_float32(vector)

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this backend and its parameters."""
        return {
            "name": self.name,
            "model": self.model,
            "dim": self.dim,
            "normalize": self.normalize,
            "ngram_range": list(self.ngram_range),
            "seed": self.seed,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
        }
