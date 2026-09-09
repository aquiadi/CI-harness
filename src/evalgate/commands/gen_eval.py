"""``evalgate gen-eval``: draft retrieval candidates for human review.

The model sees one chunk at a time and proposes questions answerable from it,
together with the sentence that answers them. Every candidate is written with
status ``draft``; nothing a model produced counts as an eval slot until a
person has accepted it in ``make label ARGS="label.mode=retrieval"``.

Two checks run before a candidate is written at all, because they are cheap and
a reviewer's attention is not: the evidence span must appear verbatim in the
chunk it was drawn from, and the question must not share a long literal phrase
with that evidence. A question that quotes its own answer measures string
matching.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console

from evalgate.chunking.base import Chunk
from evalgate.config import EvalGenConfig, load_config, resolve_path, typed_node
from evalgate.evalsets.schemas import Difficulty, Origin, RetrievalExample, ReviewStatus
from evalgate.evalsets.store import read_jsonl, write_jsonl
from evalgate.evaluation.spans import contains_span, flatten
from evalgate.models.base import ModelRequest
from evalgate.models.client import build_model_client
from evalgate.models.structured import build_tool, call_structured
from evalgate.pipeline import build_chunker, build_tokenizer, chunk_corpus, load_documents
from evalgate.prompts import load_prompt

console = Console()

TOOL_NAME = "submit_candidates"
TOOL_DESCRIPTION = "Submit candidate retrieval eval questions drawn from one chunk."
# A question sharing this many consecutive words with its evidence is testing
# string overlap rather than retrieval.
MAX_SHARED_WORDS = 5


class Candidate(BaseModel):
    """One drafted question and the sentence that answers it."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(description="A question answerable from this chunk alone.")
    evidence_span: str = Field(description="The answering sentence, copied verbatim.")
    difficulty: Difficulty = Field(description="lexical or semantic.")
    rationale: str = Field(description="One sentence on what the question tests.")


class CandidateBatch(BaseModel):
    """The tool's input: every candidate for one chunk."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[Candidate] = Field(description="Zero or more candidates; empty is valid.")


@dataclass(frozen=True, slots=True)
class Rejection:
    """A candidate discarded before it reached the reviewer."""

    chunk_id: str
    question: str
    reason: str


def shares_long_phrase(question: str, evidence: str, max_words: int = MAX_SHARED_WORDS) -> bool:
    """Whether the question copies a run of words from its own evidence."""
    question_words = flatten(question.casefold()).split()
    evidence_text = flatten(evidence.casefold())
    windows = (
        " ".join(question_words[index : index + max_words])
        for index in range(max(0, len(question_words) - max_words + 1))
    )
    return any(window in evidence_text for window in windows)


def sample_chunks(chunks: list[Chunk], count: int, min_tokens: int, seed: int) -> list[Chunk]:
    """Deterministically sample chunks, spread across documents.

    Round-robin over documents before sampling within them, so a long document
    cannot take every slot and leave a short one untested.
    """
    eligible = [chunk for chunk in chunks if chunk.n_tokens >= min_tokens]
    by_document: dict[str, list[Chunk]] = {}
    for chunk in eligible:
        by_document.setdefault(chunk.doc_id, []).append(chunk)

    rng = random.Random(seed)
    for owned in by_document.values():
        rng.shuffle(owned)

    picked: list[Chunk] = []
    while len(picked) < count and any(by_document.values()):
        for owned in by_document.values():
            if owned and len(picked) < count:
                picked.append(owned.pop())
    return sorted(picked, key=lambda chunk: chunk.id)


def run(overrides: list[str]) -> int:
    """Draft candidates over sampled chunks and merge them into the eval set."""
    cfg: DictConfig = load_config(overrides=overrides)
    settings = typed_node(cfg, "evalgen", EvalGenConfig)
    prompts_dir = resolve_path(cfg, "paths.prompts_dir")
    prompt = load_prompt(prompts_dir, settings.prompt)

    tokenizer = build_tokenizer(cfg)
    chunker = build_chunker(cfg, tokenizer)
    corpus = load_documents(cfg)
    chunks = chunk_corpus(corpus, chunker)
    selected = sample_chunks(chunks, settings.sample_chunks, settings.min_chunk_tokens, cfg.seed)

    client = build_model_client(cfg)
    tool = build_tool(TOOL_NAME, TOOL_DESCRIPTION, CandidateBatch)

    path = resolve_path(cfg, "evalsets.retrieval_path")
    existing = read_jsonl(path, RetrievalExample)
    known = {example.id for example in existing}
    drafted: list[RetrievalExample] = []
    rejected: list[Rejection] = []
    input_tokens = output_tokens = 0

    console.print(
        f"drafting from {len(selected)} of {len(chunks)} chunks with {settings.model} "
        f"(api mode {cfg.api.mode}); target {settings.target_slots} slots"
    )

    for chunk in selected:
        if len(existing) + len(drafted) >= settings.target_slots:
            break
        result = call_structured(
            client,
            ModelRequest(
                model=settings.model,
                prompt=prompt.render(
                    chunk_id=chunk.id,
                    document_title=chunk.doc_title,
                    chunk_text=chunk.text,
                    max_questions=settings.max_questions_per_chunk,
                ),
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                tool=tool,
                purpose="evalgen",
                prompt_hash=prompt.sha256,
            ),
            CandidateBatch,
            prompts_dir,
            max_attempts=settings.max_attempts,
        )
        input_tokens += result.response.input_tokens
        output_tokens += result.response.output_tokens

        for index, candidate in enumerate(
            result.value.candidates[: settings.max_questions_per_chunk]
        ):
            if not contains_span(chunk.text, candidate.evidence_span):
                rejected.append(Rejection(chunk.id, candidate.question, "evidence not in chunk"))
                continue
            if shares_long_phrase(candidate.question, candidate.evidence_span):
                rejected.append(Rejection(chunk.id, candidate.question, "question quotes evidence"))
                continue
            example_id = f"gen-{chunk.id.replace('#', '-')}-{index}"
            if example_id in known:
                continue
            drafted.append(
                RetrievalExample(
                    id=example_id,
                    question=candidate.question,
                    gold_spans=[candidate.evidence_span],
                    gold_doc_id=chunk.doc_id,
                    gold_chunk_ids=[chunk.id],
                    reference_chunker=chunker.name,
                    corpus=corpus.name,
                    corpus_hash=corpus.corpus_hash,
                    difficulty=candidate.difficulty,
                    origin=Origin.MODEL,
                    status=ReviewStatus.DRAFT,
                    notes=candidate.rationale,
                )
            )

    merged = {example.id: example for example in existing}
    merged.update({example.id: example for example in drafted})
    write_jsonl(path, [merged[key] for key in sorted(merged)])

    cost = (
        input_tokens * settings.input_usd_per_mtok + output_tokens * settings.output_usd_per_mtok
    ) / 1_000_000.0
    console.print(
        f"{len(drafted)} candidates drafted, {len(rejected)} discarded before review, "
        f"{len(merged)} slots total in {path}"
    )
    console.print(f"tokens in/out {input_tokens}/{output_tokens}, cost ${cost:.4f}")
    for rejection in rejected[:10]:
        console.print(f"  [yellow]discarded[/yellow] {rejection.chunk_id}: {rejection.reason}")
    console.print('review them with: make label ARGS="label.mode=retrieval"')
    return 0
