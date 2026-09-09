"""Matching gold evidence against retrieved text.

Gold evidence is a verbatim span of a source document (see
``evalgate.evalsets.schemas``). A chunk counts as a hit when it contains that
span, compared with whitespace collapsed on both sides -- the corpus wraps
sentences across lines, so a literal substring test would fail on evidence that
is plainly present.

Matching is case-sensitive. In regulatory text, case carries meaning ("Article"
versus "article", "Member State" versus "member state"), and a case-insensitive
match would quietly accept a chunk that merely mentions the words.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def flatten(text: str) -> str:
    """Collapse all whitespace so line wrapping cannot break a match."""
    return " ".join(text.split())


def contains_span(text: str, span: str) -> bool:
    """Whether the text contains the gold span, ignoring line wrapping."""
    return flatten(span) in flatten(text)


def contains_any(text: str, spans: Iterable[str]) -> bool:
    """Whether the text contains at least one of the gold spans."""
    return any(contains_span(text, span) for span in spans)


def first_hit_rank(texts: Sequence[str], spans: Sequence[str]) -> int | None:
    """Zero-based rank of the first text containing any gold span."""
    for rank, text in enumerate(texts):
        if contains_any(text, spans):
            return rank
    return None


def hit_ranks(texts: Sequence[str], spans: Sequence[str]) -> list[int]:
    """Every rank whose text contains a gold span."""
    return [rank for rank, text in enumerate(texts) if contains_any(text, spans)]
