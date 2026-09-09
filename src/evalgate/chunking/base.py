"""Chunk records and the shared machinery every chunker uses.

A chunk is identified by ``{doc_id}#{ordinal:04d}``. The id is document-scoped
rather than corpus-global so that adding a document does not renumber every
other document's chunks -- eval sets reference gold chunk ids, and those
references should survive a corpus addition even though the corpus hash (and
therefore run comparability) changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from evalgate.ingest.documents import Document
from evalgate.ingest.tokens import Span, TokenEstimator

CHUNK_ID_FORMAT = "{doc_id}#{ordinal:04d}"


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of text."""

    id: str
    doc_id: str
    doc_title: str
    ordinal: int
    text: str
    n_tokens: int
    char_start: int
    char_end: int
    section: str | None = None


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into chunks."""

    name: str

    def chunk(self, document: Document) -> list[Chunk]:
        """Split one document."""
        ...

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this chunker and its parameters, for the run record."""
        ...


def chunk_id(doc_id: str, ordinal: int) -> str:
    """Build a chunk id."""
    return CHUNK_ID_FORMAT.format(doc_id=doc_id, ordinal=ordinal)


def atoms_per_target(text: str, tokenizer: TokenEstimator, target_tokens: int) -> int:
    """How many tokenizer atoms approximate ``target_tokens`` tokens.

    Calibrated once per document rather than by growing a window and re-counting
    it, which would be quadratic in document length for no extra accuracy.
    """
    atoms = len(tokenizer.spans(text))
    if atoms == 0:
        return max(1, target_tokens)
    tokens = max(1, tokenizer.count(text))
    return max(1, int(target_tokens * atoms / tokens))


def window_spans(token_spans: list[Span], size: int, stride: int) -> list[Span]:
    """Fixed windows over atom spans, expressed as character spans."""
    if not token_spans:
        return []
    size = max(1, size)
    stride = max(1, stride)
    out: list[Span] = []
    for start in range(0, len(token_spans), stride):
        window = token_spans[start : start + size]
        if not window:
            break
        out.append((window[0][0], window[-1][1]))
        if start + size >= len(token_spans):
            break
    return out


def trim_span(text: str, span: Span) -> Span | None:
    """Shrink a span to exclude leading and trailing whitespace."""
    start, end = span
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def build_chunks(
    document: Document,
    spans: list[Span],
    tokenizer: TokenEstimator,
    sections: list[str | None] | None = None,
) -> list[Chunk]:
    """Turn character spans into numbered chunks, dropping empty ones."""
    if sections is not None and len(sections) != len(spans):
        raise ValueError("sections must align with spans")
    chunks: list[Chunk] = []
    for index, span in enumerate(spans):
        trimmed = trim_span(document.text, span)
        if trimmed is None:
            continue
        start, end = trimmed
        text = document.text[start:end]
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                id=chunk_id(document.id, ordinal),
                doc_id=document.id,
                doc_title=document.title,
                ordinal=ordinal,
                text=text,
                n_tokens=tokenizer.count(text),
                char_start=start,
                char_end=end,
                section=sections[index] if sections is not None else None,
            )
        )
    return chunks


def merge_undersized(
    spans: list[Span],
    text: str,
    tokenizer: TokenEstimator,
    min_tokens: int,
) -> list[Span]:
    """Fold spans shorter than ``min_tokens`` into their predecessor.

    A 12-token fragment is not a retrievable unit; it dilutes the index and
    inflates recall@k by occupying a slot with nothing in it.
    """
    if min_tokens <= 0:
        return spans
    merged: list[Span] = []
    for span in spans:
        if merged and tokenizer.count(text[span[0] : span[1]]) < min_tokens:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return merged
