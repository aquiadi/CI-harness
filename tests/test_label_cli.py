from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import rich.prompt
from omegaconf import DictConfig

import evalgate.commands.label as label_cli
from evalgate.config import load_config
from evalgate.evalsets.schemas import AnswerExample, HumanLabel, RetrievalExample, ReviewStatus
from evalgate.evalsets.store import read_jsonl, write_jsonl


def answer(example_id: str, text: str = "An answer [d#0001].") -> AnswerExample:
    return AnswerExample(
        id=example_id,
        question=f"question {example_id}?",
        reference_answer=text,
        gold_spans=["evidence"],
        gold_doc_id="syn_reg_main",
        corpus="cbam_synthetic",
        corpus_hash="h",
    )


@pytest.fixture
def workspace(tmp_path: Path, fixture_corpus_dir: Path) -> DictConfig:
    answers = tmp_path / "answers.jsonl"
    write_jsonl(answers, [answer("a-1"), answer("a-2"), answer("a-3")])
    return load_config(
        overrides=[
            "corpus.name=fixture",
            f"corpus.local_dir={fixture_corpus_dir}",
            "embedder=hashed",
            "chunker.target_tokens=96",
            f"paths.index_dir={tmp_path / 'index'}",
            f"evalsets.answers_path={answers}",
            f"evalsets.human_labels_path={tmp_path / 'labels.jsonl'}",
            "label.labeller=tester",
            "label.context_k=2",
        ]
    )


def scripted(answers: list[str]) -> Iterator[str]:
    return iter(answers)


@pytest.fixture
def keystrokes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Feed the CLI a queue of answers in place of a terminal."""
    queue: list[str] = []

    def fake_ask(prompt: str, choices: list[str] | None = None, **_: object) -> str:
        if not queue:
            raise KeyboardInterrupt
        return queue.pop(0)

    monkeypatch.setattr(rich.prompt.Prompt, "ask", staticmethod(fake_ask))
    return queue


def test_labels_are_written_as_they_are_given(
    workspace: DictConfig, keystrokes: list[str], tmp_path: Path
) -> None:
    keystrokes.extend(["5", "4", "3", "n", "2", "2", "2", "n"])
    assert label_cli.run(_overrides(workspace)) == 0
    labels = read_jsonl(tmp_path / "labels.jsonl", HumanLabel)
    assert [record.example_id for record in labels] == ["a-1", "a-2"]
    assert labels[0].scores.as_dict() == {
        "groundedness": 5,
        "relevance": 4,
        "citation_correctness": 3,
    }
    assert labels[0].labeller == "tester"


def test_interruption_keeps_everything_already_rated(
    workspace: DictConfig, keystrokes: list[str], tmp_path: Path
) -> None:
    """Ctrl-C mid-item costs that item and nothing else."""
    keystrokes.extend(["5", "5", "5", "n", "4"])  # second item interrupted mid-axis
    assert label_cli.run(_overrides(workspace)) == 0
    assert [record.example_id for record in read_jsonl(tmp_path / "labels.jsonl", HumanLabel)] == [
        "a-1"
    ]


def test_a_second_session_resumes_where_the_first_stopped(
    workspace: DictConfig, keystrokes: list[str], tmp_path: Path
) -> None:
    keystrokes.extend(["1", "1", "1", "n"])
    label_cli.run(_overrides(workspace))
    keystrokes.extend(["2", "2", "2", "n"])
    label_cli.run(_overrides(workspace))
    labels = read_jsonl(tmp_path / "labels.jsonl", HumanLabel)
    assert [record.example_id for record in labels] == ["a-1", "a-2"]


def test_quit_stops_without_writing_the_current_item(
    workspace: DictConfig, keystrokes: list[str], tmp_path: Path
) -> None:
    keystrokes.extend(["5", "q"])
    assert label_cli.run(_overrides(workspace)) == 0
    assert read_jsonl(tmp_path / "labels.jsonl", HumanLabel) == []


def test_skip_leaves_the_item_unlabelled(
    workspace: DictConfig, keystrokes: list[str], tmp_path: Path
) -> None:
    keystrokes.extend(["s", "3", "3", "3", "n"])
    assert label_cli.run(_overrides(workspace)) == 0
    assert [record.example_id for record in read_jsonl(tmp_path / "labels.jsonl", HumanLabel)] == [
        "a-2"
    ]


def test_a_note_is_recorded(workspace: DictConfig, keystrokes: list[str], tmp_path: Path) -> None:
    keystrokes.extend(["5", "5", "5", "y", "cited the wrong annex"])
    assert label_cli.run(_overrides(workspace)) == 0
    assert read_jsonl(tmp_path / "labels.jsonl", HumanLabel)[0].notes == "cited the wrong annex"


def test_retrieval_review_records_verdicts_immediately(
    workspace: DictConfig, keystrokes: list[str], tmp_path: Path
) -> None:
    path = tmp_path / "retrieval.jsonl"
    write_jsonl(
        path,
        [
            RetrievalExample(
                id=f"c-{index}",
                question="q?",
                gold_spans=["evidence"],
                gold_doc_id="d",
                corpus="c",
                corpus_hash="h",
            )
            for index in range(3)
        ],
    )
    keystrokes.extend(["a", "r"])
    overrides = [*_overrides(workspace), "label.mode=retrieval", f"evalsets.retrieval_path={path}"]
    assert label_cli.run(overrides) == 0
    statuses = [record.status for record in read_jsonl(path, RetrievalExample)]
    assert statuses == [ReviewStatus.ACCEPTED, ReviewStatus.REJECTED, ReviewStatus.DRAFT]


def test_unknown_mode_is_refused(workspace: DictConfig) -> None:
    assert label_cli.run([*_overrides(workspace), "label.mode=nonsense"]) == 2


def _overrides(cfg: DictConfig) -> list[str]:
    """Rebuild the override list the fixture composed, for re-composition."""
    return [
        f"corpus.name={cfg.corpus.name}",
        f"corpus.local_dir={cfg.corpus.local_dir}",
        "embedder=hashed",
        f"chunker.target_tokens={cfg.chunker.target_tokens}",
        f"paths.index_dir={cfg.paths.index_dir}",
        f"evalsets.answers_path={cfg.evalsets.answers_path}",
        f"evalsets.human_labels_path={cfg.evalsets.human_labels_path}",
        f"label.labeller={cfg.label.labeller}",
        f"label.context_k={cfg.label.context_k}",
    ]
