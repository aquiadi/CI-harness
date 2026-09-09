"""A rule-based judge, used as a baseline.

This exists to answer the question a sceptical reader asks first: is the LLM
judge better than a cheap rule? A judge that agrees with people no better than
string matching does is not worth its latency or its bill, and without a
baseline in the report there is no way to tell.

The rules are deliberately simple and are stated in full here, because a
baseline whose behaviour is unclear is not a baseline:

*groundedness*: the share of the answer's sentences whose content words are
covered by the retrieved context. An answer that says only what the context
says scores high; one that introduces unsupported specifics scores low.

*relevance*: overlap between the question's content words and the answer's,
with a correct refusal detected separately -- an answer that declines when the
context does not contain the gold evidence is scored as relevant, and one that
declines when the evidence is present is not.

*citation_correctness*: the share of the answer's citations that name a chunk
in its context, penalised when it makes no citations at all.

It is fast, free, deterministic, and immune by construction to the position and
self-preference biases the probes look for -- which is itself informative when
comparing it to the LLM judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from evalgate.citations import audit_citations
from evalgate.context import context_chunk_ids
from evalgate.corpus.manifest import utc_now_iso
from evalgate.evalsets.schemas import AxisScores, JudgeScore
from evalgate.evaluation.spans import contains_any, flatten
from evalgate.hashing import sha256_text
from evalgate.judging.base import JudgeInput

WORD = re.compile(r"[a-z0-9]+")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
# Words that carry no evidence either way when comparing an answer to context.
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
        "for",
        "from",
        "has",
        "have",
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
        "which",
        "will",
        "with",
        "not",
        "no",
    ]
)
REFUSAL_MARKERS = (
    "does not address",
    "does not contain",
    "cannot answer",
    "not in the context",
    "insufficient",
)


def _content_words(text: str) -> set[str]:
    return {word for word in WORD.findall(text.casefold()) if word not in STOPWORDS}


def _scale(fraction: float, low: int, high: int) -> int:
    """Map a 0..1 coverage fraction onto the configured integer scale."""
    span = high - low
    return max(low, min(high, low + round(fraction * span)))


@dataclass(slots=True)
class HeuristicJudge:
    """Rule-based baseline scoring on the same axes as the LLM judge."""

    name: str
    model: str
    axes: list[str]
    scale_min: int
    scale_max: int

    def score(self, item: JudgeInput) -> JudgeScore:
        """Grade one answer by rule."""
        context_text = " ".join(result.text for result in item.context)
        context_words = _content_words(context_text)
        answer_words = _content_words(item.answer)
        refused = any(marker in item.answer.casefold() for marker in REFUSAL_MARKERS)
        evidence_present = bool(item.gold_spans) and contains_any(context_text, item.gold_spans)

        grounded = self._groundedness(item.answer, context_words)
        relevance = self._relevance(item.question, answer_words, refused, evidence_present)
        citations = self._citations(item)

        scores = {
            "groundedness": grounded,
            "relevance": relevance,
            "citation_correctness": citations,
        }
        return JudgeScore(
            example_id=item.example_id,
            answer_hash=sha256_text(item.answer),
            scores=AxisScores(**{axis: scores[axis] for axis in self.axes}),
            rationales=dict.fromkeys(self.axes, "rule-based baseline"),
            judge_model=self.model,
            prompt_hash="",
            scored_at=utc_now_iso(),
            variant=item.variant,
            generator=item.generator,
        )

    def _groundedness(self, answer: str, context_words: set[str]) -> int:
        sentences = [part for part in SENTENCE.split(flatten(answer)) if part.strip()]
        if not sentences:
            return self.scale_min
        covered = []
        for sentence in sentences:
            words = _content_words(sentence)
            covered.append(len(words & context_words) / len(words) if words else 1.0)
        return _scale(sum(covered) / len(covered), self.scale_min, self.scale_max)

    def _relevance(
        self, question: str, answer_words: set[str], refused: bool, evidence_present: bool
    ) -> int:
        if refused:
            # Declining is right when the evidence is absent and wrong when it
            # is there; the score has to distinguish the two.
            return self.scale_max if not evidence_present else self.scale_min
        question_words = _content_words(question)
        if not question_words:
            return self.scale_min
        overlap = len(question_words & answer_words) / len(question_words)
        return _scale(overlap, self.scale_min, self.scale_max)

    def _citations(self, item: JudgeInput) -> int:
        audit = audit_citations(item.answer, context_chunk_ids(item.context))
        if audit.count == 0:
            return self.scale_min
        return _scale(audit.precision, self.scale_min, self.scale_max)

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this judge and its parameters."""
        return {
            "name": self.name,
            "model": self.model,
            "axes": list(self.axes),
            "scale": [self.scale_min, self.scale_max],
        }
