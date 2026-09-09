#!/usr/bin/env python
"""Materialise the hand-written seed eval sets into JSONL.

    make seed

Reads data/eval/seeds/*.yaml, verifies that every gold span really occurs in
the document it names, resolves the chunk ids that hold each span under the
reference chunker, substitutes citation placeholders in the seed answers, and
merges the result into the eval sets without disturbing records that came from
anywhere else.

The verification is the point. A gold span with a typo produces an eval slot
that no retriever can ever satisfy, which shows up later as an unexplained
recall ceiling. Failing here instead is cheap.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml
from rich.console import Console

from evalgate.chunking.base import Chunk
from evalgate.config import load_config, resolve_path
from evalgate.errors import EvalgateError
from evalgate.evalsets.schemas import (
    AnswerExample,
    AxisScores,
    Difficulty,
    HumanLabel,
    Origin,
    RetrievalExample,
    ReviewStatus,
)
from evalgate.evalsets.store import read_jsonl, write_jsonl
from evalgate.evaluation.spans import contains_span
from evalgate.hashing import sha256_text
from evalgate.pipeline import build_chunker, build_tokenizer, chunk_corpus, load_documents

console = Console()

SEED_LABELLER = "seed-author"
SEED_LABELS_FILE = "seed_labels.jsonl"
PLACEHOLDER_GOLD = "${gold}"
PLACEHOLDER_WRONG = "${wrong}"
PLACEHOLDER_MISSING = "${missing}"
MISSING_CHUNK_ID = "{doc}#9999"


class SeedError(EvalgateError):
    """Raised when a seed file does not match the corpus."""


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SeedError(f"{path}: expected a mapping at the top level")
    return payload


def _chunks_for(chunks: list[Chunk], doc_id: str) -> list[Chunk]:
    owned = [chunk for chunk in chunks if chunk.doc_id == doc_id]
    if not owned:
        raise SeedError(f"no chunks for document {doc_id!r}; is it in the corpus?")
    return owned


def _resolve_spans(chunks: list[Chunk], doc_id: str, spans: list[str], where: str) -> list[str]:
    """Verify each span occurs in the document and return the chunks holding it."""
    owned = _chunks_for(chunks, doc_id)
    document_text = "\n".join(chunk.text for chunk in owned)
    hits: list[str] = []
    for span in spans:
        if not contains_span(document_text, span):
            raise SeedError(
                f"{where}: gold span not found in {doc_id}.\n  span: {span[:120]}\n"
                "  Fix the span, or the document, so the two agree."
            )
        holders = [chunk.id for chunk in owned if contains_span(chunk.text, span)]
        # A span can straddle a chunk boundary under some chunkers; that is a
        # real property of the chunking, not a seed error, so it is recorded
        # rather than raised. The metrics match on spans, not on these ids.
        hits.extend(holders)
    return sorted(dict.fromkeys(hits))


def _seed_retrieval(
    seed: dict[str, Any], chunks: list[Chunk], corpus_name: str, corpus_hash: str, chunker: str
) -> list[RetrievalExample]:
    examples: list[RetrievalExample] = []
    for item in seed["examples"]:
        gold_ids = _resolve_spans(chunks, item["doc"], item["spans"], item["id"])
        examples.append(
            RetrievalExample(
                id=item["id"],
                question=item["question"],
                gold_spans=item["spans"],
                gold_doc_id=item["doc"],
                gold_chunk_ids=gold_ids,
                reference_chunker=chunker,
                corpus=corpus_name,
                corpus_hash=corpus_hash,
                difficulty=Difficulty(item.get("difficulty", "semantic")),
                origin=Origin(seed.get("origin", "seed")),
                status=ReviewStatus(seed.get("status", "accepted")),
                notes=item.get("notes"),
            )
        )
    return examples


def _substitute(answer: str, gold_ids: list[str], other_id: str, doc_id: str) -> str:
    """Replace citation placeholders with real chunk ids."""
    gold = gold_ids[0] if gold_ids else MISSING_CHUNK_ID.format(doc=doc_id)
    return (
        answer.replace(PLACEHOLDER_GOLD, gold)
        .replace(PLACEHOLDER_WRONG, other_id)
        .replace(PLACEHOLDER_MISSING, MISSING_CHUNK_ID.format(doc=doc_id))
        .strip()
    )


def _seed_answers(
    seed: dict[str, Any], chunks: list[Chunk], corpus_name: str, corpus_hash: str
) -> tuple[list[AnswerExample], list[HumanLabel]]:
    labelled_at = str(seed["labelled_at"])
    examples: list[AnswerExample] = []
    labels: list[HumanLabel] = []
    for item in seed["examples"]:
        doc_id = item["doc"]
        gold_ids = _resolve_spans(chunks, doc_id, item["spans"], item["id"])
        elsewhere = next(chunk.id for chunk in chunks if chunk.doc_id != doc_id)
        answer = _substitute(item["answer"], gold_ids, elsewhere, doc_id)
        examples.append(
            AnswerExample(
                id=item["id"],
                question=item["question"],
                reference_answer=answer,
                gold_spans=item["spans"],
                gold_doc_id=doc_id,
                corpus=corpus_name,
                corpus_hash=corpus_hash,
                origin=Origin.SEED,
                status=ReviewStatus.ACCEPTED,
                notes=f"flaw: {item.get('flaw', 'none')}",
            )
        )
        scores = item.get("author_scores")
        if scores:
            labels.append(
                HumanLabel(
                    example_id=item["id"],
                    answer_hash=sha256_text(answer),
                    scores=AxisScores(**scores),
                    notes=f"authored alongside the answer; flaw: {item.get('flaw', 'none')}",
                    labeller=SEED_LABELLER,
                    labelled_at=labelled_at,
                )
            )
    return examples, labels


class _Identified(Protocol):
    """Anything with a string id, which is every eval record."""

    id: str


def _merge[T: _Identified](existing: Sequence[T], fresh: Sequence[T]) -> list[T]:
    """Replace records whose id matches, keep everything else, sort by id.

    Re-seeding must not disturb reviewed candidates that came from the model.
    """
    by_id: dict[str, T] = {str(record.id): record for record in existing}
    for record in fresh:
        by_id[str(record.id)] = record
    return [by_id[key] for key in sorted(by_id)]


def main(argv: list[str] | None = None) -> int:
    """Materialise the seed files."""
    overrides = list(argv if argv is not None else sys.argv[1:])
    cfg = load_config(overrides=["corpus=cbam_synthetic", *overrides])

    tokenizer = build_tokenizer(cfg)
    chunker = build_chunker(cfg, tokenizer)
    corpus = load_documents(cfg)
    chunks = chunk_corpus(corpus, chunker)
    console.print(
        f"corpus {corpus.name} ({corpus.size} documents, {len(chunks)} chunks under {chunker.name})"
    )

    seeds_dir = resolve_path(cfg, "paths.eval_dir") / "seeds"
    retrieval_seed = _load_yaml(seeds_dir / "retrieval_seed.yaml")
    answers_seed = _load_yaml(seeds_dir / "answers_seed.yaml")

    retrieval = _seed_retrieval(
        retrieval_seed, chunks, corpus.name, corpus.corpus_hash, chunker.name
    )
    answers, labels = _seed_answers(answers_seed, chunks, corpus.name, corpus.corpus_hash)

    retrieval_path = resolve_path(cfg, "evalsets.retrieval_path")
    answers_path = resolve_path(cfg, "evalsets.answers_path")
    labels_path = resolve_path(cfg, "paths.eval_dir") / SEED_LABELS_FILE

    write_jsonl(retrieval_path, _merge(read_jsonl(retrieval_path, RetrievalExample), retrieval))
    write_jsonl(answers_path, _merge(read_jsonl(answers_path, AnswerExample), answers))
    write_jsonl(labels_path, labels)

    console.print(f"{len(retrieval)} retrieval slots -> {retrieval_path}")
    console.print(f"{len(answers)} answer examples -> {answers_path}")
    console.print(f"{len(labels)} seed labels -> {labels_path} (attributed to {SEED_LABELLER})")
    unresolved = [example.id for example in retrieval if not example.gold_chunk_ids]
    if unresolved:
        console.print(
            f"[yellow]{len(unresolved)} spans straddle a chunk boundary under "
            f"{chunker.name}: {', '.join(unresolved)}[/yellow]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
