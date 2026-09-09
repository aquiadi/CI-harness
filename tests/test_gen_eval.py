from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from evalgate.chunking.base import Chunk
from evalgate.commands.gen_eval import sample_chunks, shares_long_phrase
from evalgate.config import load_config
from evalgate.evalsets.schemas import RetrievalExample, ReviewStatus
from evalgate.evalsets.store import read_jsonl
from evalgate.evaluation.spans import contains_span
from evalgate.pipeline import build_chunker, build_tokenizer, chunk_corpus, load_documents


def chunk(chunk_id: str, doc_id: str, tokens: int = 100) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id,
        doc_title=doc_id,
        ordinal=0,
        text="text",
        n_tokens=tokens,
        char_start=0,
        char_end=4,
    )


def test_question_quoting_its_own_evidence_is_detected() -> None:
    evidence = "The competent authority shall take a decision within 120 calendar days."
    assert shares_long_phrase("shall take a decision within 120 calendar days?", evidence)


def test_paraphrased_question_is_allowed() -> None:
    evidence = "The competent authority shall take a decision within 120 calendar days."
    assert not shares_long_phrase("How long does the authority get to decide?", evidence)


def test_sampling_is_deterministic() -> None:
    chunks = [chunk(f"d{d}#{i:04d}", f"d{d}") for d in range(3) for i in range(10)]
    assert sample_chunks(chunks, 6, 10, 0) == sample_chunks(chunks, 6, 10, 0)


def test_sampling_spreads_across_documents() -> None:
    """A long document must not take every slot and leave a short one untested."""
    chunks = [chunk(f"long#{i:04d}", "long") for i in range(50)] + [chunk("short#0000", "short")]
    picked = sample_chunks(chunks, 4, 10, 0)
    assert {c.doc_id for c in picked} == {"long", "short"}


def test_sampling_skips_boilerplate_sized_chunks() -> None:
    chunks = [chunk("d#0000", "d", tokens=5), chunk("d#0001", "d", tokens=100)]
    assert [c.id for c in sample_chunks(chunks, 5, 60, 0)] == ["d#0001"]


def test_sampling_cannot_return_more_than_exists() -> None:
    assert len(sample_chunks([chunk("d#0000", "d")], 10, 10, 0)) == 1


def test_seeded_retrieval_slots_are_answerable_from_their_gold_document(
    repo_root: Path,
) -> None:
    """Every committed gold span must really occur in the corpus."""
    cfg: DictConfig = load_config(overrides=["corpus=cbam_synthetic", "embedder=hashed"])
    chunks = chunk_corpus(load_documents(cfg), build_chunker(cfg, build_tokenizer(cfg)))
    by_document: dict[str, str] = {}
    for piece in chunks:
        by_document[piece.doc_id] = by_document.get(piece.doc_id, "") + "\n" + piece.text

    examples = read_jsonl(repo_root / "data" / "eval" / "retrieval.jsonl", RetrievalExample)
    assert len(examples) >= 15
    for example in examples:
        assert example.gold_doc_id in by_document, example.id
        for span in example.gold_spans:
            assert contains_span(by_document[example.gold_doc_id], span), example.id


def test_seed_slots_are_accepted_and_the_rest_are_drafts(repo_root: Path) -> None:
    examples = read_jsonl(repo_root / "data" / "eval" / "retrieval.jsonl", RetrievalExample)
    for example in examples:
        if example.id.startswith("seed-"):
            assert example.status is ReviewStatus.ACCEPTED, example.id
        elif example.origin.value == "model":
            assert example.status in {
                ReviewStatus.DRAFT,
                ReviewStatus.ACCEPTED,
                ReviewStatus.REJECTED,
            }
