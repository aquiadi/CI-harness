"""Section-aware chunking for legal instruments.

CBAM documents are articles, annexes and numbered paragraphs. A chunk that
spans an article boundary mixes two obligations, which is exactly the failure
mode that produces confidently wrong compliance answers. So headings are hard
boundaries: no chunk ever contains two article-level headings.

Undersized sections fold *forwards*, into the section that follows them. A
chapter title with no body of its own ("CHAPTER II / Obligations of the
reporting declarant") belongs with the first article underneath it, not with
the last article of the previous chapter -- and carrying that context into the
chunk is worth more at retrieval time than dropping it. Oversized sections are
split internally into target-sized windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from evalgate.chunking.base import (
    Chunk,
    atoms_per_target,
    build_chunks,
    trim_span,
    window_spans,
)
from evalgate.ingest.documents import Document
from evalgate.ingest.tokens import Span, TokenEstimator


@dataclass(frozen=True, slots=True)
class SectionAwareChunker:
    """Split on headings first; subdivide only what is too large."""

    name: str
    target_tokens: int
    overlap_tokens: int
    min_tokens: int
    max_tokens: int
    heading_patterns: list[str]
    tokenizer: TokenEstimator
    _headings: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Compile the heading alternation once; the dataclass is frozen."""
        combined = "|".join(f"(?:{pattern})" for pattern in self.heading_patterns)
        object.__setattr__(self, "_headings", re.compile(combined, re.MULTILINE))

    def chunk(self, document: Document) -> list[Chunk]:
        """Split one document at heading boundaries."""
        text = document.text
        sections = self._sections(text)
        spans: list[Span] = []
        labels: list[str | None] = []
        for span, label in sections:
            for piece in self._subdivide(text, span):
                spans.append(piece)
                labels.append(label)
        spans, labels = self._fold_undersized(text, spans, labels)
        return build_chunks(document, spans, self.tokenizer, labels)

    def _sections(self, text: str) -> list[tuple[Span, str | None]]:
        """Partition the document at heading starts."""
        matches = list(self._headings.finditer(text))
        if not matches:
            return [((0, len(text)), None)]

        sections: list[tuple[Span, str | None]] = []
        if matches[0].start() > 0:
            sections.append(((0, matches[0].start()), None))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections.append(((match.start(), end), _heading_label(text, match.start())))
        return sections

    def _subdivide(self, text: str, span: Span) -> list[Span]:
        """Split a section that exceeds max_tokens into target-sized windows."""
        body = text[span[0] : span[1]]
        if self.tokenizer.count(body) <= self.max_tokens:
            return [span]
        size = atoms_per_target(body, self.tokenizer, self.target_tokens)
        overlap = atoms_per_target(body, self.tokenizer, self.overlap_tokens)
        stride = max(1, size - min(overlap, size - 1))
        local = window_spans(self.tokenizer.spans(body), size, stride)
        return [(span[0] + start, span[0] + end) for start, end in local] or [span]

    def _fold_undersized(
        self, text: str, spans: list[Span], labels: list[str | None]
    ) -> tuple[list[Span], list[str | None]]:
        """Fold sub-minimum sections forward into the section that follows.

        Headings, recital numbers and one-line cross-references are common in
        these documents and are not independently retrievable. Folding forward
        keeps a container heading with the provision it introduces; the label
        taken is the destination's, so a chunk is still named by the article it
        contains. A trailing undersized section has nothing to fold into and
        joins its predecessor instead.
        """
        out_spans: list[Span] = []
        out_labels: list[str | None] = []
        pending: Span | None = None

        for span, label in zip(spans, labels, strict=True):
            start = pending[0] if pending is not None else span[0]
            candidate = (start, span[1])
            trimmed = trim_span(text, candidate)
            if trimmed is None or self.tokenizer.count(text[candidate[0] : candidate[1]]) < (
                self.min_tokens
            ):
                pending = candidate
                continue
            pending = None
            out_spans.append(candidate)
            out_labels.append(label)

        if pending is not None:
            if out_spans:
                out_spans[-1] = (out_spans[-1][0], pending[1])
            else:
                out_spans.append(pending)
                out_labels.append(labels[-1] if labels else None)
        return out_spans, out_labels

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this chunker and its parameters."""
        return {
            "name": self.name,
            "target_tokens": self.target_tokens,
            "overlap_tokens": self.overlap_tokens,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "heading_patterns": list(self.heading_patterns),
            "tokenizer": self.tokenizer.fingerprint(),
        }


def _heading_label(text: str, start: int) -> str:
    """The heading line itself, used as the chunk's section label.

    A heading pattern may match leading whitespace, so skip forward to the
    first non-space character before reading the line -- otherwise the label
    is the empty string for every heading preceded by a blank line.
    """
    while start < len(text) and text[start].isspace():
        start += 1
    end = text.find("\n", start)
    return " ".join(text[start : end if end != -1 else len(text)].split())
