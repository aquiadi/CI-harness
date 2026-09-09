from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig
from scripts.seed_evalsets import SeedError, _resolve_spans, _substitute, main

from evalgate.chunking.base import Chunk
from evalgate.config import load_config
from evalgate.evalsets.schemas import AnswerExample, HumanLabel
from evalgate.evalsets.store import read_jsonl
from evalgate.evaluation.spans import contains_span
from evalgate.hashing import sha256_text
from evalgate.pipeline import build_chunker, build_tokenizer, chunk_corpus, load_documents


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    cfg: DictConfig = load_config(overrides=["corpus=cbam_synthetic", "embedder=hashed"])
    return chunk_corpus(load_documents(cfg), build_chunker(cfg, build_tokenizer(cfg)))


def test_a_mistyped_span_fails_loudly(chunks: list[Chunk]) -> None:
    """A gold span with a typo makes a slot no retriever can ever satisfy."""
    with pytest.raises(SeedError, match="gold span not found"):
        _resolve_spans(chunks, "syn_reg_main", ["this sentence is not in the corpus"], "test")


def test_a_real_span_resolves_to_the_chunks_holding_it(chunks: list[Chunk]) -> None:
    ids = _resolve_spans(
        chunks,
        "syn_reg_main",
        ["shall take a decision within 120 calendar days of receipt of a complete application"],
        "test",
    )
    assert ids
    assert all(chunk_id.startswith("syn_reg_main#") for chunk_id in ids)


def test_an_unknown_document_fails(chunks: list[Chunk]) -> None:
    with pytest.raises(SeedError, match="no chunks for document"):
        _resolve_spans(chunks, "not_a_document", ["x"], "test")


def test_placeholders_are_substituted() -> None:
    answer = _substitute("a [${gold}] b [${wrong}] c [${missing}]", ["d#0001"], "other#0002", "d")
    assert "[d#0001]" in answer
    assert "[other#0002]" in answer
    assert "[d#9999]" in answer
    assert "${" not in answer


def test_placeholders_fall_back_when_no_gold_chunk_resolved() -> None:
    assert "[d#9999]" in _substitute("a [${gold}]", [], "other#0002", "d")


def test_seed_answers_cite_chunks_that_exist_except_where_intended(repo_root: Path) -> None:
    examples = read_jsonl(repo_root / "data" / "eval" / "answers.jsonl", AnswerExample)
    assert len(examples) >= 15
    deliberate = {"seed-a-013"}  # written to cite a nonexistent chunk
    for example in examples:
        if example.id in deliberate:
            assert "#9999" in example.reference_answer
        else:
            assert "#9999" not in example.reference_answer, example.id


def test_seed_answers_gold_spans_are_real(repo_root: Path, chunks: list[Chunk]) -> None:
    examples = read_jsonl(repo_root / "data" / "eval" / "answers.jsonl", AnswerExample)
    text_by_doc: dict[str, str] = {}
    for chunk in chunks:
        text_by_doc[chunk.doc_id] = text_by_doc.get(chunk.doc_id, "") + "\n" + chunk.text
    for example in examples:
        assert example.gold_doc_id is not None
        for span in example.gold_spans:
            assert contains_span(text_by_doc[example.gold_doc_id], span), example.id


def test_seed_labels_are_attributed_and_pinned_to_their_answer(repo_root: Path) -> None:
    """Seed labels are not independent human labels and must say whose they are."""
    labels = read_jsonl(repo_root / "data" / "eval" / "seed_labels.jsonl", HumanLabel)
    answers = {
        example.id: example
        for example in read_jsonl(repo_root / "data" / "eval" / "answers.jsonl", AnswerExample)
    }
    assert labels
    for record in labels:
        assert record.labeller == "seed-author"
        assert record.answer_hash == sha256_text(answers[record.example_id].reference_answer)


def test_seeding_is_byte_idempotent(repo_root: Path) -> None:
    """Re-seeding must not dirty the working tree, so no wall-clock anywhere."""
    files = ["answers.jsonl", "retrieval.jsonl", "seed_labels.jsonl"]
    before = {
        name: (repo_root / "data" / "eval" / name).read_text(encoding="utf-8") for name in files
    }
    assert main([]) == 0
    for name in files:
        assert (repo_root / "data" / "eval" / name).read_text(encoding="utf-8") == before[name]
