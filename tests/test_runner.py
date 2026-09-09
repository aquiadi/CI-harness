from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from evalgate.config import load_config
from evalgate.evalsets.schemas import AnswerExample, RetrievalExample, ReviewStatus
from evalgate.evalsets.store import write_jsonl
from evalgate.evaluation.runner import TASK_ANSWER, TASK_RETRIEVAL, RunOutcome, evaluate
from evalgate.generation.factory import build_generator
from evalgate.judging.factory import build_judge
from evalgate.pipeline import build_stack
from evalgate.prompts import load_prompt


@pytest.fixture
def workspace(tmp_path: Path, fixture_corpus_dir: Path) -> DictConfig:
    retrieval = tmp_path / "retrieval.jsonl"
    answers = tmp_path / "answers.jsonl"
    write_jsonl(
        retrieval,
        [
            RetrievalExample(
                id="r-1",
                question="When must the quarterly report be submitted?",
                gold_spans=["no later than one month after the end of that quarter"],
                gold_doc_id="fixture_reg_a",
                corpus="fixture",
                corpus_hash="h",
                status=ReviewStatus.ACCEPTED,
            ),
            RetrievalExample(
                id="r-2",
                question="Nothing in this corpus answers this question at all",
                gold_spans=["a span that does not appear anywhere"],
                gold_doc_id="fixture_reg_a",
                corpus="fixture",
                corpus_hash="h",
                status=ReviewStatus.ACCEPTED,
            ),
            RetrievalExample(
                id="r-3",
                question="A draft that should not be scored",
                gold_spans=["Article 3"],
                gold_doc_id="fixture_reg_a",
                corpus="fixture",
                corpus_hash="h",
                status=ReviewStatus.DRAFT,
            ),
        ],
    )
    write_jsonl(
        answers,
        [
            AnswerExample(
                id="a-1",
                question="When must the quarterly report be submitted?",
                reference_answer="unused: the generator produces the answer",
                gold_spans=["no later than one month after the end of that quarter"],
                gold_doc_id="fixture_reg_a",
                corpus="fixture",
                corpus_hash="h",
            )
        ],
    )
    return load_config(
        overrides=[
            "corpus.name=fixture",
            f"corpus.local_dir={fixture_corpus_dir}",
            "embedder=hashed",
            "generator=extractive",
            "judge=heuristic",
            "retriever=bm25",
            "retriever.k=3",
            "chunker.target_tokens=96",
            f"paths.index_dir={tmp_path / 'index'}",
            f"evalsets.retrieval_path={retrieval}",
            f"evalsets.answers_path={answers}",
        ]
    )


def outcome(cfg: DictConfig, repo_root: Path) -> RunOutcome:
    stack = build_stack(cfg)
    prompts = {"generator": load_prompt(repo_root / "prompts", cfg.generator.prompt)}
    return evaluate(
        cfg,
        stack,
        build_generator(cfg, stack.tokenizer),
        build_judge(cfg),
        prompts,
    )


def test_both_eval_sets_are_measured(workspace: DictConfig, repo_root: Path) -> None:
    result = outcome(workspace, repo_root)
    assert (result.rows["task"] == TASK_RETRIEVAL).sum() == 2
    assert (result.rows["task"] == TASK_ANSWER).sum() == 1


def test_only_accepted_retrieval_slots_are_scored(workspace: DictConfig, repo_root: Path) -> None:
    """A draft candidate nobody has reviewed must not silently enter the metrics."""
    result = outcome(workspace, repo_root)
    scored = set(result.rows[result.rows["task"] == TASK_RETRIEVAL]["example_id"])
    assert scored == {"r-1", "r-2"}


def test_an_unanswerable_slot_scores_zero_not_missing(
    workspace: DictConfig, repo_root: Path
) -> None:
    result = outcome(workspace, repo_root)
    row = result.rows[result.rows["example_id"] == "r-2"].iloc[0]
    assert row["recall_at_k"] == 0.0
    assert row["first_hit_rank"] == -1
    assert row["reciprocal_rank"] == 0.0


def test_metrics_carry_the_aggregate_shape(workspace: DictConfig, repo_root: Path) -> None:
    metrics = outcome(workspace, repo_root).metrics
    for key in (
        "recall_at_k",
        "mrr",
        "ndcg_at_10",
        "p50_latency_s",
        "p95_latency_s",
        "cost_per_query_usd",
        "projected_cost_per_query_usd",
        "composite_quality",
    ):
        assert key in metrics, key
    assert 0.0 <= metrics["composite_quality"] <= 1.0


def test_serving_latency_excludes_judging(workspace: DictConfig, repo_root: Path) -> None:
    """The Pareto latency axis is what a user waits for, not what evaluation costs."""
    row = outcome(workspace, repo_root).rows.query("task == 'answer'").iloc[0]
    assert row["latency_s"] == pytest.approx(
        row["retrieval_latency_s"] + row["generation_latency_s"]
    )


def test_projected_cost_is_priced_at_the_reference_model(
    workspace: DictConfig, repo_root: Path
) -> None:
    """A generator that makes no API call still has a cost if it ran on the API."""
    row = outcome(workspace, repo_root).rows.query("task == 'answer'").iloc[0]
    assert row["cost_usd"] == 0.0
    assert row["projected_cost_usd"] > 0.0


def test_meta_records_every_hash_needed_to_compare_runs(
    workspace: DictConfig, repo_root: Path
) -> None:
    meta = outcome(workspace, repo_root).meta
    assert meta.corpus_hash
    assert meta.index_hash
    assert meta.prompt_hashes
    assert set(meta.evalset_hashes) == {"retrieval", "answers"}
    assert meta.fingerprint["retriever"]["name"] == "bm25"


def test_eval_set_content_changes_the_recorded_hash(
    workspace: DictConfig, repo_root: Path, tmp_path: Path
) -> None:
    before = outcome(workspace, repo_root).meta.evalset_hashes["answers"]
    write_jsonl(
        tmp_path / "answers.jsonl",
        [
            AnswerExample(
                id="a-1",
                question="A different question entirely?",
                reference_answer="x",
                gold_doc_id="fixture_reg_a",
                corpus="fixture",
                corpus_hash="h",
            )
        ],
    )
    assert outcome(workspace, repo_root).meta.evalset_hashes["answers"] != before


def test_the_run_is_reproducible(workspace: DictConfig, repo_root: Path) -> None:
    first = outcome(workspace, repo_root)
    second = outcome(workspace, repo_root)
    assert first.metrics["composite_quality"] == second.metrics["composite_quality"]
    assert first.metrics["recall_at_k"] == second.metrics["recall_at_k"]
    assert list(first.rows["answer"].dropna()) == list(second.rows["answer"].dropna())
