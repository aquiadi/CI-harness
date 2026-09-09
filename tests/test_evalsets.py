from __future__ import annotations

from pathlib import Path

import pytest

from evalgate.evalsets.schemas import (
    AnswerExample,
    AxisScores,
    Difficulty,
    HumanLabel,
    Origin,
    RetrievalExample,
    ReviewStatus,
)
from evalgate.evalsets.store import (
    EvalStoreError,
    append_jsonl,
    index_by,
    read_jsonl,
    write_jsonl,
)


def example(example_id: str = "x-1") -> RetrievalExample:
    return RetrievalExample(
        id=example_id,
        question="When is the report due?",
        gold_spans=["no later than one month after the end of the quarter"],
        gold_doc_id="doc",
        corpus="c",
        corpus_hash="h",
    )


def label(example_id: str = "x-1") -> HumanLabel:
    return HumanLabel(
        example_id=example_id,
        answer_hash="a" * 64,
        scores=AxisScores(groundedness=4, relevance=5, citation_correctness=3),
        labeller="tester",
        labelled_at="2026-01-01T00:00:00Z",
    )


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.jsonl"
    write_jsonl(path, [example("a"), example("b")])
    assert [record.id for record in read_jsonl(path, RetrievalExample)] == ["a", "b"]


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "absent.jsonl", RetrievalExample) == []


def test_append_is_durable_and_incremental(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    append_jsonl(path, label("a"))
    assert len(read_jsonl(path, HumanLabel)) == 1
    append_jsonl(path, label("b"))
    assert [record.example_id for record in read_jsonl(path, HumanLabel)] == ["a", "b"]


def test_blank_lines_are_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    append_jsonl(path, label("a"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert len(read_jsonl(path, HumanLabel)) == 1


def test_a_bad_line_names_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    append_jsonl(path, label("a"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"example_id": "b"}\n')
    with pytest.raises(EvalStoreError, match=r":2:"):
        read_jsonl(path, HumanLabel)


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.jsonl"
    write_jsonl(path, [example()])
    assert list(tmp_path.iterdir()) == [path]


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"extra_forbidden|Extra inputs"):
        RetrievalExample.model_validate(
            {
                "id": "x",
                "question": "q",
                "gold_spans": ["s"],
                "gold_doc_id": "d",
                "corpus": "c",
                "corpus_hash": "h",
                "surprise": 1,
            }
        )


def test_gold_spans_cannot_be_empty() -> None:
    with pytest.raises(ValueError, match=r"at least 1 item|too_short"):
        RetrievalExample.model_validate(
            {
                "id": "x",
                "question": "q",
                "gold_spans": [],
                "gold_doc_id": "d",
                "corpus": "c",
                "corpus_hash": "h",
            }
        )


def test_defaults_are_draft_and_model_origin() -> None:
    record = example()
    assert record.status is ReviewStatus.DRAFT
    assert record.origin is Origin.MODEL
    assert record.difficulty is Difficulty.SEMANTIC


def test_index_by_takes_the_last_write() -> None:
    first, second = example("a"), example("a")
    second.question = "changed"
    assert index_by([first, second], "id")["a"].question == "changed"


def test_answer_example_requires_a_reference_answer() -> None:
    with pytest.raises(ValueError, match="reference_answer"):
        AnswerExample.model_validate(
            {"id": "x", "question": "q", "corpus": "c", "corpus_hash": "h"}
        )


def test_axis_scores_expose_a_dict() -> None:
    assert AxisScores(groundedness=1, relevance=2, citation_correctness=3).as_dict() == {
        "groundedness": 1,
        "relevance": 2,
        "citation_correctness": 3,
    }
