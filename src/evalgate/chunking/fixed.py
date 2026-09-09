"""Fixed-token chunking: the baseline every other strategy is judged against."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalgate.chunking.base import Chunk, atoms_per_target, build_chunks, window_spans
from evalgate.ingest.documents import Document
from evalgate.ingest.tokens import TokenEstimator


@dataclass(frozen=True, slots=True)
class FixedTokenChunker:
    """Sliding window of a fixed token width, ignoring document structure."""

    name: str
    target_tokens: int
    overlap_tokens: int
    tokenizer: TokenEstimator

    def chunk(self, document: Document) -> list[Chunk]:
        """Split into equal-width overlapping windows."""
        text = document.text
        size = atoms_per_target(text, self.tokenizer, self.target_tokens)
        overlap = atoms_per_target(text, self.tokenizer, self.overlap_tokens)
        stride = max(1, size - min(overlap, size - 1))
        spans = window_spans(self.tokenizer.spans(text), size, stride)
        return build_chunks(document, spans, self.tokenizer)

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this chunker and its parameters."""
        return {
            "name": self.name,
            "target_tokens": self.target_tokens,
            "overlap_tokens": self.overlap_tokens,
            "tokenizer": self.tokenizer.fingerprint(),
        }
