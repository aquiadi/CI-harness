from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from evalgate.config import load_config
from evalgate.evalsets.judgments import JudgmentsMeta, scores_path, write_judgments
from evalgate.evalsets.schemas import AnswerExample, AxisScores, HumanLabel, JudgeScore
from evalgate.evalsets.store import write_jsonl
from evalgate.hashing import sha256_text
from evalgate.ingest.tokens import RegexTokenEstimator
from evalgate.reporting.calibration import build_report, partition_labels, write_report

TOKENIZER = RegexTokenEstimator(name="t", pattern=r"\w+|[^\w\s]", tokens_per_word=1.3)


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
    example_id: str,
    text: str,
    scores: tuple[int, int, int],
    variant: str = "primary",
    model: str = "test-judge",
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
        judge_model=model,
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
            f"evalsets.judgments_dir={tmp_path / 'judgments'}",
        ]
    )


def write_scores(
    tmp_path: Path, judgments: list[JudgeScore], meta: JudgmentsMeta | None = None
) -> None:
    """Scores under the judge's own file, with provenance when given."""
    directory = tmp_path / "judgments"
    if meta is not None:
        write_judgments(directory, meta, judgments)
        return
    model = judgments[0].judge_model if judgments else "test-judge"
    write_jsonl(scores_path(directory, model), judgments)


def provenance(model: str = "test-judge", provider: str = "api") -> JudgmentsMeta:
    return JudgmentsMeta(
        judge_model=model,
        judge_provider=provider,
        judge_prompt="judge/rubric_v1.md",
        judge_prompt_hash="f" * 64,
        axes=["groundedness", "relevance", "citation_correctness"],
        scale_min=1,
        scale_max=5,
        answers_from="reference",
        generators=["reference"],
        api_mode="live",
        stack={"embedder": "local / bge", "retriever": "hybrid (k=10)"},
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
    write_scores(tmp_path, judgments)


def test_report_reports_agreement(workspace: DictConfig, tmp_path: Path) -> None:
    populate(tmp_path)
    report = build_report(workspace, TOKENIZER)
    assert "Judge calibration" in report
    assert "Agreement with a-person labels" in report
    assert "kappa" in report


def test_report_says_so_when_there_are_no_judgments(workspace: DictConfig, tmp_path: Path) -> None:
    write_jsonl(tmp_path / "answers.jsonl", [answer("e-1", "text")])
    report = build_report(workspace, TOKENIZER)
    assert "No judgments on disk" in report


def test_report_says_so_when_there_are_no_labels(workspace: DictConfig, tmp_path: Path) -> None:
    write_jsonl(tmp_path / "answers.jsonl", [answer("e-1", "text")])
    write_scores(tmp_path, [judgment("e-1", "text", (5, 5, 5))])
    report = build_report(workspace, TOKENIZER)
    assert "No labels on disk" in report


def test_seed_labels_carry_the_independence_warning(workspace: DictConfig, tmp_path: Path) -> None:
    populate(tmp_path, labeller="seed-author")
    write_jsonl(tmp_path / "seed_labels.jsonl", [])
    report = build_report(workspace, TOKENIZER)
    assert "not independent human labels" in report


def test_human_labels_do_not_carry_the_warning(workspace: DictConfig, tmp_path: Path) -> None:
    populate(tmp_path, labeller="a-person")
    report = build_report(workspace, TOKENIZER)
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
    write_scores(tmp_path, [judgment("e-1", "a rewritten answer", (1, 1, 1))])
    report = build_report(workspace, TOKENIZER)
    assert "no overlapping labelled items" in report


def test_position_probe_reports_when_swapped_scores_exist(
    workspace: DictConfig, tmp_path: Path
) -> None:
    populate(tmp_path, swapped=True)
    report = build_report(workspace, TOKENIZER)
    assert "identical score" in report


def test_position_probe_says_it_did_not_run_without_swapped_scores(
    workspace: DictConfig, tmp_path: Path
) -> None:
    populate(tmp_path, swapped=False)
    report = build_report(workspace, TOKENIZER)
    assert "no swapped-context judgments" in report


def test_self_preference_probe_explains_what_it_needs(
    workspace: DictConfig, tmp_path: Path
) -> None:
    populate(tmp_path)
    report = build_report(workspace, TOKENIZER)
    assert "answers to the same questions from two generators" in report


def test_worst_disagreements_are_listed(workspace: DictConfig, tmp_path: Path) -> None:
    populate(tmp_path, judge_offset=4)
    report = build_report(workspace, TOKENIZER)
    assert "Worst disagreements" in report
    assert "| e-0 | groundedness | 5 | 1 | 4 |" in report


def test_report_is_a_pure_function_of_its_inputs(workspace: DictConfig, tmp_path: Path) -> None:
    """No timestamp: regenerating an unchanged report must not dirty the tree."""
    populate(tmp_path)
    first = build_report(workspace, TOKENIZER)
    second = build_report(workspace, TOKENIZER)
    assert first == second


def test_rejudging_the_same_answers_does_not_change_the_report(
    workspace: DictConfig, tmp_path: Path
) -> None:
    """Re-running the judge produces identical scores with a new timestamp.

    That must not show up as a report diff, or every regenerated report trains
    the reader to ignore changes in reports/.
    """
    populate(tmp_path)
    before = build_report(workspace, TOKENIZER)
    rejudged = [
        judgment(f"e-{index}", f"answer number {index} [d#0001]", (5, 4, 3)) for index in range(5)
    ]
    for record in rejudged:
        record.scored_at = "2099-12-31T23:59:59Z"
    write_scores(tmp_path, rejudged)
    assert build_report(workspace, TOKENIZER) == before


def test_write_report_reports_whether_it_changed(tmp_path: Path) -> None:
    path = tmp_path / "r.md"
    assert write_report(path, "content") is True
    assert write_report(path, "content") is False
    assert write_report(path, "other") is True


def test_the_header_names_the_judge_that_produced_the_scores(
    workspace: DictConfig, tmp_path: Path
) -> None:
    """Rendered under a heuristic config, an LLM judge's scores stay the LLM's.

    The report used to describe the judge and the retrieval stack from the
    configuration it was rendered under, and the default profile attributed an
    LLM judge's kappa to the rule-based judge and a bge index to a hashed one.
    """
    populate(tmp_path)
    texts = {f"e-{index}": f"answer number {index} [d#0001]" for index in range(5)}
    write_scores(
        tmp_path,
        [judgment(key, text, (5, 4, 3)) for key, text in texts.items()],
        provenance(),
    )
    report = build_report(workspace, TOKENIZER)
    assert workspace.judge.model == "rule-based-v1"
    assert "rule-based-v1" not in report
    assert "judge: `test-judge` (provider `api`)" in report
    assert "embedder: local / bge" in report
    assert "hashed" not in report


def test_scores_without_provenance_say_so(workspace: DictConfig, tmp_path: Path) -> None:
    populate(tmp_path)
    report = build_report(workspace, TOKENIZER)
    assert "retrieval stack: not recorded" in report
    assert "provider not recorded" in report


def test_two_judges_are_reported_side_by_side_rule_last(
    workspace: DictConfig, tmp_path: Path
) -> None:
    """The rule-based judge is the baseline an LLM judge has to beat."""
    populate(tmp_path)
    texts = {f"e-{index}": f"answer number {index} [d#0001]" for index in range(5)}
    write_scores(
        tmp_path,
        [judgment(key, text, (1, 4, 3), model="rule-based-v1") for key, text in texts.items()],
        provenance("rule-based-v1", "heuristic"),
    )
    write_scores(
        tmp_path,
        [judgment(key, text, (5, 4, 3), model="a-model") for key, text in texts.items()],
        provenance("a-model"),
    )
    report = build_report(workspace, TOKENIZER)
    assert "Judges compared against a-person labels" in report
    llm = report.index("## Judge: `a-model`")
    rule = report.index("## Judge: `rule-based-v1`")
    assert llm < rule
