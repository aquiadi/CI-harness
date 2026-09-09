"""Request and response shapes for the serving layer.

The response carries the retrieval trace and the cost alongside the answer, not
because a caller always wants them but because a RAG answer without them cannot
be debugged. When an answer is wrong, the first question is always "what did it
retrieve", and an API that cannot say has to be reproduced offline to find out.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """One question for the system."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=100, description="Override the configured k.")


class RetrievedChunk(BaseModel):
    """One chunk the retriever returned."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str | None
    rank: int
    score: float


class Citation(BaseModel):
    """One chunk id the answer cited, and whether the context contained it."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    valid: bool


class RetrievalTrace(BaseModel):
    """What retrieval did, in enough detail to reproduce it."""

    model_config = ConfigDict(extra="forbid")

    retriever: str
    k: int
    index_hash: str
    chunks: list[RetrievedChunk]
    latency_s: float


class Cost(BaseModel):
    """What answering cost, measured and projected."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int
    output_tokens: int
    context_tokens: int
    usd: float
    projected_usd: float


class Latency(BaseModel):
    """Where the time went."""

    model_config = ConfigDict(extra="forbid")

    retrieval_s: float
    generation_s: float
    total_s: float


class Provenance(BaseModel):
    """What produced this answer, for the reader who does not trust it."""

    model_config = ConfigDict(extra="forbid")

    corpus: str
    corpus_hash: str
    index_hash: str
    generator_model: str
    generator_prompt_hash: str
    config_hash: str
    replayed: bool


class QueryResponse(BaseModel):
    """The answer and everything behind it."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation]
    retrieval: RetrievalTrace
    cost: Cost
    latency: Latency
    provenance: Provenance


class Health(BaseModel):
    """Readiness, and what the process is serving."""

    model_config = ConfigDict(extra="forbid")

    status: str
    corpus: str
    corpus_hash: str
    index_hash: str
    chunks: int
    retriever: str
    generator_model: str
