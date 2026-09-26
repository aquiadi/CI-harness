"""`make judge` writes one file per judge and never replaces another's scores.

Every judge used to write `judge_scores.jsonl`. The calibration report, finding
no judgments, told the reader to run the rule-based judge -- which would have
overwritten the only LLM judgments on disk with a rule's.
"""

from __future__ import annotations

import json
from pathlib import Path

from evalgate.commands.judge import run
from evalgate.evalsets.schemas import AnswerExample, JudgeScore
from evalgate.evalsets.store import read_jsonl, write_jsonl


def _overrides(tmp_path: Path, fixture_corpus_dir: Path) -> list[str]:
    return [
        "corpus.name=fixture",
        f"corpus.local_dir={fixture_corpus_dir}",
        "embedder=hashed",
        "judge=heuristic",
        f"paths.index_dir={tmp_path / 'index'}",
        f"evalsets.answers_path={tmp_path / 'answers.jsonl'}",
        f"evalsets.judgments_dir={tmp_path / 'judgments'}",
    ]


def _answers(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "answers.jsonl",
        [
            AnswerExample(
                id="a-1",
                question="When must the quarterly report be submitted?",
                reference_answer="No later than one month after the end of that quarter.",
                gold_spans=["no later than one month after the end of that quarter"],
                gold_doc_id="fixture_reg_a",
                corpus="fixture",
                corpus_hash="h",
            )
        ],
    )


def test_a_second_judge_leaves_the_first_judges_scores_alone(
    tmp_path: Path, fixture_corpus_dir: Path
) -> None:
    _answers(tmp_path)
    judgments = tmp_path / "judgments"
    judgments.mkdir()
    existing = judgments / "qwen_qwen3.8-27b.jsonl"
    existing.write_text('{"placeholder": "an LLM judge\'s scores"}\n', encoding="utf-8")

    code = run(_overrides(tmp_path, fixture_corpus_dir))

    assert code == 0
    assert existing.read_text(encoding="utf-8") == '{"placeholder": "an LLM judge\'s scores"}\n'
    meta = json.loads((judgments / "rule-based-v1.meta.json").read_text(encoding="utf-8"))
    assert meta["judge_provider"] == "heuristic"
    assert meta["stack"]["embedder"].startswith("hashed")
    assert meta["stack"]["corpus"].startswith("fixture")
    scores = read_jsonl(judgments / "rule-based-v1.jsonl", JudgeScore)
    assert len(scores) == 2, "primary and swapped-context"


def test_grading_another_generator_adds_to_the_judges_scores(
    tmp_path: Path, fixture_corpus_dir: Path
) -> None:
    """The self-preference probe needs one judge's scores for two generators."""
    _answers(tmp_path)
    assert run(_overrides(tmp_path, fixture_corpus_dir)) == 0
    path = tmp_path / "judgments" / "rule-based-v1.jsonl"
    first = read_jsonl(path, JudgeScore)
    for record in first:
        record.generator = "another-generator"
    write_jsonl(path, first)

    assert run(_overrides(tmp_path, fixture_corpus_dir)) == 0
    generators = {record.generator for record in read_jsonl(path, JudgeScore)}
    assert generators == {"another-generator", "reference"}
