"""Rendering retrieved chunks as prompt context.

One formatter, used by the generator, the judge and the labelling CLI, so that
the text the judge grades is byte-identical to the text the generator saw. Two
formatters would eventually disagree about whitespace or ordering, and the
judge would then be grading groundedness against a context the answer never
had.
"""

from __future__ import annotations

from collections.abc import Sequence

from evalgate.retrieval.base import Retrieved

BLOCK = "[{chunk_id}] ({doc_title})\n{text}"
SEPARATOR = "\n\n"


def format_context(results: Sequence[Retrieved]) -> str:
    """Render retrieved chunks, each labelled with the id an answer must cite."""
    return SEPARATOR.join(
        BLOCK.format(chunk_id=result.chunk_id, doc_title=result.doc_title, text=result.text)
        for result in results
    )


def context_chunk_ids(results: Sequence[Retrieved]) -> list[str]:
    """The chunk ids present in the rendered context, in order."""
    return [result.chunk_id for result in results]
