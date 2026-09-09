"""An extractive generator: the honest floor for a RAG system.

It answers by quoting the sentences of the top-ranked chunks that overlap the
question, with the citation attached. No model, no cost, no network, and a
latency dominated entirely by retrieval.

It exists for two reasons. First, it is the baseline any generator has to beat:
if a model's answers do not score better than "quote the best-matching
sentences", the model is not adding anything and the ablation should show that.
Second, it lets the whole measurement path -- ablation sweep, Pareto frontier,
regression gate -- run end to end with no credentials, which is the same
property that makes pull-request CI free.

Its answers are stilted and it cannot synthesise across chunks. That is the
point of having something better to compare it against.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from evalgate.context import format_context
from evalgate.generation.base import Generated, GenerationInput
from evalgate.ingest.tokens import TokenEstimator

WORD = re.compile(r"[a-z0-9]+")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "shall",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "not",
        "no",
        "you",
        "your",
        "we",
        "our",
    ]
)
NO_ANSWER = "The retrieved context does not contain enough to answer this question."


def _content_words(text: str) -> set[str]:
    return {word for word in WORD.findall(text.casefold()) if word not in STOPWORDS}


@dataclass(slots=True)
class ExtractiveGenerator:
    """Quotes the best-matching sentences from the top chunks, with citations."""

    name: str
    model: str
    tokenizer: TokenEstimator
    max_sentences: int
    max_chunks: int
    min_overlap: int

    def generate(self, item: GenerationInput) -> Generated:
        """Answer by quotation."""
        started = time.perf_counter()
        question_words = _content_words(item.question)
        scored: list[tuple[int, int, str, str]] = []

        for rank, result in enumerate(item.context[: self.max_chunks]):
            for sentence in SENTENCE.split(" ".join(result.text.split())):
                overlap = len(_content_words(sentence) & question_words)
                if overlap >= self.min_overlap:
                    # Rank breaks ties towards the better-retrieved chunk, and
                    # the sentence text makes the order total and reproducible.
                    scored.append((-overlap, rank, sentence, result.chunk_id))

        scored.sort()
        chosen = scored[: self.max_sentences]
        if chosen:
            answer = " ".join(f"{sentence} [{chunk_id}]" for _, _, sentence, chunk_id in chosen)
        else:
            answer = NO_ANSWER

        context = format_context(item.context)
        return Generated(
            example_id=item.example_id,
            answer=answer,
            model=self.model,
            input_tokens=0,
            output_tokens=0,
            latency_s=time.perf_counter() - started,
            context_tokens=self.tokenizer.count(context),
        )

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this generator and its parameters."""
        return {
            "name": self.name,
            "model": self.model,
            "max_sentences": self.max_sentences,
            "max_chunks": self.max_chunks,
            "min_overlap": self.min_overlap,
        }
