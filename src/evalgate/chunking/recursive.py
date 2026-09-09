"""Recursive structural chunking.

Split on the largest structural boundary that works -- blank lines, then line
breaks, then sentences, then words -- and only fall back to a blind token
window when no separator does. Then pack the resulting fragments greedily back
up to the target size, so a chunk ends at a structural boundary rather than
mid-sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from evalgate.chunking.base import (
    Chunk,
    atoms_per_target,
    build_chunks,
    merge_undersized,
    window_spans,
)
from evalgate.ingest.documents import Document
from evalgate.ingest.tokens import Span, TokenEstimator


@dataclass(frozen=True, slots=True)
class RecursiveStructuralChunker:
    """Split on structural separators, then repack to the target size."""

    name: str
    target_tokens: int
    overlap_tokens: int
    min_tokens: int
    separators: list[str]
    tokenizer: TokenEstimator

    def chunk(self, document: Document) -> list[Chunk]:
        """Split one document."""
        text = document.text
        atoms = atoms_per_target(text, self.tokenizer, self.target_tokens)
        fragments = self._split(text, (0, len(text)), list(self.separators), atoms)
        packed = self._pack(text, fragments)
        packed = merge_undersized(packed, text, self.tokenizer, self.min_tokens)
        packed = self._apply_overlap(text, packed)
        return build_chunks(document, packed, self.tokenizer)

    def _fits(self, text: str, span: Span) -> bool:
        return self.tokenizer.count(text[span[0] : span[1]]) <= self.target_tokens

    def _split(self, text: str, span: Span, separators: list[str], atoms: int) -> list[Span]:
        """Recursively divide a span until each piece fits the target."""
        if self._fits(text, span):
            return [span]

        start, end = span
        for index, separator in enumerate(separators):
            pieces = self._split_on(text, span, separator)
            if len(pieces) > 1:
                rest = separators[index + 1 :]
                out: list[Span] = []
                for piece in pieces:
                    out.extend(self._split(text, piece, rest, atoms))
                return out

        # Nothing structural to cut on: a blind window is the last resort.
        spans = self.tokenizer.spans(text[start:end])
        shifted = [(start + s, start + e) for s, e in spans]
        return window_spans(shifted, atoms, atoms) or [span]

    @staticmethod
    def _split_on(text: str, span: Span, separator: str) -> list[Span]:
        """Split a span on a literal separator, keeping character offsets."""
        start, end = span
        pieces: list[Span] = []
        cursor = start
        while cursor < end:
            found = text.find(separator, cursor, end)
            if found == -1:
                break
            stop = found + len(separator)
            if stop > cursor:
                pieces.append((cursor, stop))
            cursor = stop
        if cursor < end:
            pieces.append((cursor, end))
        return [piece for piece in pieces if piece[1] > piece[0]]

    def _pack(self, text: str, fragments: list[Span]) -> list[Span]:
        """Greedily merge adjacent fragments up to the target size."""
        packed: list[Span] = []
        for fragment in fragments:
            if not packed:
                packed.append(fragment)
                continue
            candidate = (packed[-1][0], fragment[1])
            if self._fits(text, candidate):
                packed[-1] = candidate
            else:
                packed.append(fragment)
        return packed

    def _apply_overlap(self, text: str, spans: list[Span]) -> list[Span]:
        """Extend each chunk backwards to repeat the tail of its predecessor."""
        if self.overlap_tokens <= 0 or len(spans) < 2:
            return spans
        atom_spans = self.tokenizer.spans(text)
        if not atom_spans:
            return spans
        back = atoms_per_target(text, self.tokenizer, self.overlap_tokens)
        starts = [span[0] for span in atom_spans]
        out = [spans[0]]
        for previous, current in pairwise(spans):
            index = _first_at_or_after(starts, current[0])
            new_start = atom_spans[max(0, index - back)][0]
            out.append((max(previous[0], new_start), current[1]))
        return out

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this chunker and its parameters."""
        return {
            "name": self.name,
            "target_tokens": self.target_tokens,
            "overlap_tokens": self.overlap_tokens,
            "min_tokens": self.min_tokens,
            "separators": list(self.separators),
            "tokenizer": self.tokenizer.fingerprint(),
        }


def _first_at_or_after(starts: list[int], position: int) -> int:
    """Index of the first atom starting at or after ``position``."""
    low, high = 0, len(starts)
    while low < high:
        middle = (low + high) // 2
        if starts[middle] < position:
            low = middle + 1
        else:
            high = middle
    return low
