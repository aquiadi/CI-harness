from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from evalgate.config import load_config
from evalgate.evalsets.schemas import AnswerExample, AxisScores, HumanLabel, JudgeScore
from evalgate.evalsets.store import write_jsonl
from evalgate.hashing import sha256_text
from evalgate.ingest.tokens import RegexTokenEstimator
from evalgate.reporting.calibration import build_report, partition_labels, write_report

TOKENIZER = RegexTokenEstimator(name="t", pattern=r"\w+|[^\w\s]", tokens_per_word=1.3)
SUMMARY = {"corpus": "fixture (abc)", "api mode": "replay"}


def answer(example_id: str, text: str) -> AnswerExample:
    return AnswerExample(
        id=example_id,
        question="q?",
        reference_answer=text,
        gold_spans=["evidence"],
        gold_doc_id="d",
        corpus="c",
        corpus_hash="h",
    )


def label(example_id: str, text: str, scores: tuple[int, int, int], labeller: str) -> HumanLabel:
    return HumanLabel(
        example_id=example_id,
        answer_hash=sha256_text(text),
        scores=AxisScores(
            groundedness=scores[0], relevance=scores[1], citation_correctness=scores[2]
        ),
        labeller=labeller,
        labelled_at="2026-01-01T00:00:00Z",
    )


def judgment(
    example_id: str, text: str, scores: tuple[int, int, int], variant: str = "primary"
) -> JudgeScore:
    return JudgeScore(
        example_id=example_id,
        answer_hash=sha256_text(text),
        scores=AxisScores(
            groundedness=scores[0], relevance=scores[1], citation_correctness=scores[2]
        ),
        rationales={
            "groundedness": "because",
            "relevance": "because",
            "citation_correctness": "because",
        },
        judge_model="test-judge",
        prompt_hash="f" * 64,
        scored_at="2026-01-01T00:00:00Z",
        variant=variant,
    )


@pytest.fixture
def workspace(tmp_path: Path, fixture_corpus_dir: Path) -> DictConfig:
    return load_config(
        overrides=[
            "corpus.name=fixture",
            f"corpus.local_dir={fixture_corpus_dir}",
            "embedder=hashed",
            "judge=heuristic",
            f"paths.index_dir={tmp_path / 'index'}",
            f"evalsets.answers_path={tmp_path / 'answers.jsonl'}",
            f"evalsets.human_labels_path={tmp_path / 'labels.jsonl'}",
            f"evalsets.seed_labels_path={tmp_path / 'seed_labels.jsonl'}",
            f"evalsets.judge_scores_path={tmp_path / 'judge.jsonl'}",
        ]
    )


def populate(
    tmp_path: Path,
    *,
    labeller: str = "a-person",
    judge_offset: int = 0,
    swapped: bool = False,
) -> None:
    texts = {f"e-{index}": f"answer number {index} [d#0001]" for index in range(5)}
    write_jsonl(tmp_path / "answers.jsonl", [answer(key, text) for key, text in texts.items()])
    write_jsonl(
        tmp_path / "labels.jsonl",
        [label(key, text, (5, 4, 3), labeller) for key, text in texts.items()],
    )
    judgments = [judgment(key, text, (5 - judge_offset, 4, 3)) for key, text in texts.items()]
    if swapped:
        judgments += [
            judgment(key, text, (5 - judge_offset, 4, 3), variant="swapped")
            for key, text in texts.items()
        ]
    write_jsonl(tmp_path / "judge.jsonl", judgments)


def test_report_reports_agreement(workspace: DictConfig, tmp_path: Path) -> None:
    populate(tmp_path)
    report = build_report(workspace, TOKENIZER, SUMMARY)
    assert "Judge calibration" in report
    assert "Agreement with a-person labels" in report
    assert "kappa" in report


def test_report_says_so_when_there_are_no_judgments(workspace: DictConfig, tmp_path: Path) -> None:
    write_jsonl(tmp_path / "answers.jsonl", [answer("e-1", "text")])
    report = build_report(workspace, TOKENIZER, SUMMARY)
    assert "No judgments on disk" in report


def test_report_says_so_when_there_are_no_labels(workspace: DictConfig, tmp_path: Path) -> None:
    write_jsonl(tmp_path / "answers.jsonl", [answer("e-1", "text")])
    write_jsonl(tmp_path / "judge.jsonl", [judgment("e-1", "text", (5, 5, 5))])
    report = build_report(workspace, TOKENIZER, SUMMARY)
    assert "No labels on disk" in report


def test_seed_labels_carry_the_independence_warning(workspace: DictConfig, tmp_path: Path) -> None:
    populate(tmp_path, labeller="seed-author")
    write_jsonl(tmp_path / "seed_labels.jsonl", [])
    report = build_report(workspace, TOKENIZER, SUMMARY)
    assert "not independent human labels" in report


def test_human_labels_do_not_carry_the_warning(workspace: DictConfig, tmp_path: Path) -> None:
    populate(tmp_path, labeller="a-person")
    report = build_report(workspace, TOKENIZER, SUMMARY)
    assert "not independent human labels" not in report


def test_label_sources_are_never_pooled() -> None:
    sets = partition_labels(
        [
            label("e-1", "t", (5, 5, 5), "a-person"),
            label("e-2", "t", (1, 1, 1), "seed-author"),
        ]
    )
    assert [item.independent for item in sets] == [True, False]
    assert [len(item.labels) for item in sets] == [1, 1]


def test_a_judgment_for_a_different_answer_is_never_paired(
    workspace: DictConfig, tmp_path: Path
) -> None:
    """Same question, different answer: pairing them would invent agreement."""
    write_jsonl(tmp_path / "answers.jsonl", [answer("e-1", "the original answer")])
    write_jsonl(tmp_path / "labels.jsonl", [label("e-1", "the original answer", (5, 5, 5), "p")])
    write_jsonl(tmp_path / "judge.jsonl", [judgment("e-1", "a rewritten answer", (1, 1, 1))])
    report = build_report(workspace, TOKENIZER, SUMMARY)
    assert "no overlapping labelled items" in report


def test_position_probe_reports_when_swapped_scores_exist(
    workspace: DictConfig, tmp_path: Path
) -> None:
    populate(tmp_path, swapped=True)
    report = build_report(workspace, TOKENIZER, SUMMARY)
    assert "identical score" in report


def test_position_probe_says_it_did_not_run_without_swapped_scores(
    workspace: DictConfig, tmp_path: Path
) -> None:
    populate(tmp_path, swapped=False)
    report = build_report(workspace, TOKENIZER, SUMMARY)
    assert "no swapped-context judgments" in report


def test_self_preference_probe_explains_what_it_needs(
    workspace: DictConfig, tmp_path: Path
) -> None:
    populate(tmp_path)
    report = build_report(workspace, TOKENIZER, SUMMARY)
    assert "needs answers to the same questions from two" in report


def test_worst_disagreements_are_listed(workspace: DictConfig, tmp_path: Path) -> None:
    populate(tmp_path, judge_offset=4)
    report = build_report(workspace, TOKENIZER, SUMMARY)
    assert "Worst disagreements" in report
    assert "| e-0 | groundedness | 5 | 1 | 4 |" in report


def test_report_is_a_pure_function_of_its_inputs(workspace: DictConfig, tmp_path: Path) -> None:
    """No timestamp: regenerating an unchanged report must not dirty the tree."""
    populate(tmp_path)
    first = build_report(workspace, TOKENIZER, SUMMARY)
    second = build_report(workspace, TOKENIZER, SUMMARY)
    assert first == second


def test_write_report_reports_whether_it_changed(tmp_path: Path) -> None:
    path = tmp_path / "r.md"
    assert write_report(path, "content") is True
    assert write_report(path, "content") is False
    assert write_report(path, "other") is True
