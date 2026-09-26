"""Binding an eval set to the corpus it was written for.

Gold evidence is a verbatim span of a source document (D-0015), so an eval set
means something only against the corpus its spans were copied from. Run the
synthetic corpus's questions against the real regulation and nothing fails:
the spans simply never match, recall reads as near zero, and that zero looks
exactly like a measurement of a bad retriever. The evalset hash does not catch
it either -- the questions are the same questions, they are just being asked
of the wrong documents.

So before anything is measured, every example must name the corpus being
measured and every gold span must occur in it. An empty eval set is refused
too: a run that scored nothing has no metrics, and the gate treats missing
metrics as a failure anyway, but "there are no accepted questions for this
corpus" is the actionable sentence, and it belongs at the start.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from evalgate.corpus.loading import LoadedCorpus
from evalgate.errors import EvalgateError
from evalgate.evalsets.schemas import AnswerExample, RetrievalExample
from evalgate.evaluation.spans import flatten

SHOWN = 5


class EvalSetCorpusError(EvalgateError):
    """Raised when an eval set cannot be scored against the configured corpus."""


def _listing(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:SHOWN])
    return shown + (f" and {len(ids) - SHOWN} more" if len(ids) > SHOWN else "")


def missing_spans(
    corpus: LoadedCorpus, examples: Sequence[RetrievalExample | AnswerExample]
) -> list[str]:
    """Ids of examples with a gold span the corpus does not contain."""
    texts = [flatten(document.text) for document in corpus.documents]
    offenders: list[str] = []
    for example in examples:
        for span in example.gold_spans:
            needle = flatten(span)
            if not any(needle in text for text in texts):
                offenders.append(example.id)
                break
    return offenders


def check_bound(
    corpus: LoadedCorpus,
    retrieval: Sequence[RetrievalExample],
    answers: Sequence[AnswerExample],
    retrieval_path: Path,
    answers_path: Path,
) -> None:
    """Refuse to measure an eval set against a corpus it was not written for."""
    if not retrieval:
        raise EvalSetCorpusError(
            f"no accepted retrieval examples for corpus {corpus.name!r} in {retrieval_path}. "
            "Draft candidates with `make gen-eval` and accept them with "
            "`make label ARGS=label.mode=retrieval`; an unreviewed draft is never scored."
        )
    if not answers:
        raise EvalSetCorpusError(
            f"no answer examples for corpus {corpus.name!r} in {answers_path}; the composite "
            "quality score needs judged answers."
        )

    examples: list[RetrievalExample | AnswerExample] = [*retrieval, *answers]
    foreign = [example.id for example in examples if example.corpus != corpus.name]
    if foreign:
        named = sorted({example.corpus for example in examples} - {corpus.name})
        raise EvalSetCorpusError(
            f"eval examples written for corpus {', '.join(map(repr, named))} cannot be scored "
            f"against {corpus.name!r}: {_listing(foreign)}. Each corpus has its own eval set "
            "under data/eval/<corpus>/."
        )

    absent = missing_spans(corpus, examples)
    if absent:
        raise EvalSetCorpusError(
            f"gold evidence not found in corpus {corpus.name!r} for {_listing(absent)}. A "
            "span that is not in the documents can never be retrieved, so every score "
            "involving it would be a silent zero. Fix the span or the example."
        )
