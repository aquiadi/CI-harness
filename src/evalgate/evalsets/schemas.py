"""Eval set records.

Gold evidence is stored as verbatim spans of the source document, not as chunk
ids. Chunk ids are a property of the chunker, and the whole point of the
ablation matrix is to vary the chunker: an eval set keyed on one chunker's ids
could not score any other. A span is a property of the corpus, so the same eval
set scores every configuration. ``gold_chunk_ids`` is still recorded, for the
reference chunking named alongside it, because it is what a human reviewer
actually looked at -- but the metrics are computed from the spans.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ReviewStatus(StrEnum):
    """Where a candidate is in the human review process."""

    DRAFT = "draft"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Difficulty(StrEnum):
    """Whether answering requires resolving a paraphrase."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"


class Origin(StrEnum):
    """Where a record came from."""

    SEED = "seed"
    MODEL = "model"
    HUMAN = "human"


class RetrievalExample(BaseModel):
    """One retrieval eval slot: a question and the evidence that answers it."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    id: str
    question: str
    gold_spans: list[str] = Field(min_length=1)
    gold_doc_id: str
    gold_chunk_ids: list[str] = Field(default_factory=list)
    reference_chunker: str | None = None
    corpus: str
    corpus_hash: str
    difficulty: Difficulty = Difficulty.SEMANTIC
    origin: Origin = Origin.MODEL
    status: ReviewStatus = ReviewStatus.DRAFT
    notes: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None


class AnswerExample(BaseModel):
    """One question and a reference answer, for judge validation."""

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    id: str
    question: str
    reference_answer: str
    gold_spans: list[str] = Field(default_factory=list)
    gold_doc_id: str | None = None
    corpus: str
    corpus_hash: str
    origin: Origin = Origin.SEED
    status: ReviewStatus = ReviewStatus.DRAFT
    notes: str | None = None


class AxisScores(BaseModel):
    """The three axes, on the configured scale."""

    model_config = ConfigDict(extra="forbid")

    groundedness: int
    relevance: int
    citation_correctness: int

    def as_dict(self) -> dict[str, int]:
        """Axis name to score."""
        return {
            "groundedness": self.groundedness,
            "relevance": self.relevance,
            "citation_correctness": self.citation_correctness,
        }


class HumanLabel(BaseModel):
    """One human rating of one answer.

    ``answer_hash`` pins which answer text was rated. A judge score is only
    comparable to a human label when both refer to the same answer, and an
    answer changes whenever the generator, the prompt or the retrieval changes.
    """

    model_config = ConfigDict(extra="forbid")

    example_id: str
    answer_hash: str
    scores: AxisScores
    notes: str | None = None
    labeller: str
    labelled_at: str
    seconds_spent: float | None = None


class JudgeScore(BaseModel):
    """One judge rating of one answer, in the same shape as a human label."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    answer_hash: str
    scores: AxisScores
    rationales: dict[str, str] = Field(default_factory=dict)
    judge_model: str
    prompt_hash: str
    scored_at: str
    attempts: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    replayed: bool = False
    # Set when the same answer was scored with the context order swapped, for
    # the position-bias probe.
    variant: str = "primary"
