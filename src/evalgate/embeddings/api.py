"""API embeddings, behind a config flag.

Not the default: every ablation would then cost money and depend on a network
round trip, which is precisely the property that stops ablations from being
run. Enabled with ``embedder=api``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from evalgate.embeddings.base import Vectors, as_float32, l2_normalise
from evalgate.embeddings.local import BackendUnavailableError

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


@dataclass(slots=True)
class VoyageEmbedder:
    """Voyage AI embeddings over HTTP."""

    name: str
    model: str
    dim: int
    batch_size: int
    normalize: bool
    query_prefix: str
    document_prefix: str
    api_key_env: str
    endpoint: str
    timeout_s: float
    usd_per_mtok: float
    max_attempts: int = 3

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise BackendUnavailableError(
                f"embedder=api needs {self.api_key_env} in the environment"
            )
        return key

    def _post(self, texts: list[str], input_type: str) -> list[list[float]]:
        payload = {"input": texts, "model": self.model, "input_type": input_type}
        headers = {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}
        last = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = httpx.post(
                    self.endpoint, json=payload, headers=headers, timeout=self.timeout_s
                )
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.is_success:
                    body: dict[str, Any] = response.json()
                    rows = sorted(body["data"], key=lambda row: int(row["index"]))
                    return [list(map(float, row["embedding"])) for row in rows]
                last = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code not in _RETRYABLE_STATUS:
                    break
            if attempt < self.max_attempts:
                time.sleep(float(2 ** (attempt - 1)))
        raise BackendUnavailableError(f"{self.model}: {last}")

    def _embed(self, texts: Sequence[str], input_type: str) -> Vectors:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            rows.extend(self._post(list(texts[start : start + self.batch_size]), input_type))
        matrix = np.asarray(rows, dtype=np.float32).reshape(len(texts), -1)
        if matrix.shape[1] != self.dim:
            raise BackendUnavailableError(
                f"{self.model} produced dim {matrix.shape[1]}, config says {self.dim}"
            )
        return l2_normalise(matrix) if self.normalize else as_float32(matrix)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """Embed passages for indexing."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._embed([self.document_prefix + text for text in texts], "document")

    def embed_query(self, text: str) -> Vectors:
        """Embed one query."""
        return as_float32(self._embed([self.query_prefix + text], "query")[0])

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this backend and its parameters."""
        return {
            "name": self.name,
            "model": self.model,
            "dim": self.dim,
            "normalize": self.normalize,
            "endpoint": self.endpoint,
        }
