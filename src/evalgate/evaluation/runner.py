"""One evaluation run: retrieval metrics, answers, judgments, cost, latency.

A run measures two eval sets against one configuration. The retrieval set gives
recall@k, MRR and nDCG@10 from gold spans. The answer set is generated over the
same stack and graded by the judge, giving the rubric axes. Both feed one
composite quality number, defined by the weights in `configs/gate/`.

Two splits are worth being explicit about, because they decide what the Pareto
frontier means.

*Serving latency excludes judging.* `latency_s` on an answer row is retrieval
plus generation: what a user waits for. Judging is measurement, not service,
and its latency is recorded in its own column.

*Cost per query excludes judging too.* The judge is an evaluation cost paid by
the team, not a serving cost paid per query. Judge spend is aggregated
separately as `eval_cost_usd`.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from evalgate.config import (
    GateConfig,
    GeneratorConfig,
    JudgeConfig,
    PricingConfig,
    TimingConfig,
    config_hash,
    resolve_path,
    typed_node,
)
from evalgate.corpus.loading import LoadedCorpus
from evalgate.corpus.manifest import utc_now_iso
from evalgate.evalsets.binding import check_bound
from evalgate.evalsets.schemas import AnswerExample, JudgeScore, RetrievalExample, ReviewStatus
from evalgate.evalsets.store import read_jsonl
from evalgate.evaluation.metrics import (
    composite_quality,
    mean,
    ndcg_at_k,
    normalise_score,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from evalgate.evaluation.spans import first_hit_rank
from evalgate.generation.base import GenerationInput, Generator
from evalgate.hashing import hash_obj, sha256_text
from evalgate.judging.base import Judge, JudgeInput
from evalgate.pipeline import RetrievalStack
from evalgate.prompts import Prompt
from evalgate.retrieval.base import Retrieved
from evalgate.runs.store import RunMeta, make_run_id

TASK_RETRIEVAL = "retrieval"
TASK_ANSWER = "answer"
NDCG_K = 10
USD_PER_MTOK = 1_000_000.0


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Everything one run produced."""

    meta: RunMeta
    rows: pd.DataFrame
    metrics: dict[str, Any]


def timed_retrieve(
    stack: RetrievalStack, question: str, k: int, repeats: int
) -> tuple[list[Retrieved], float]:
    """Retrieve, timing it `repeats` times and keeping the median.

    A single timing of a ten-millisecond retrieval is dominated by whatever
    else the machine was doing: identical back-to-back runs put p95 30% apart,
    which a 20% latency threshold cannot tell from a regression. The median of
    five, after a warm-up pass, holds it to about 10%. Retrieval is
    deterministic, so the repeats return the same chunks; the first result is
    the one scored.
    """
    timings: list[float] = []
    results: list[Retrieved] | None = None
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        found = stack.retriever.retrieve(question, k=k)
        timings.append(time.perf_counter() - started)
        if results is None:
            results = found
    return results or [], float(statistics.median(timings))


def _retrieval_rows(
    stack: RetrievalStack, examples: list[RetrievalExample], k: int, repeats: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        results, elapsed = timed_retrieve(stack, example.question, k, repeats)
        hit = first_hit_rank([result.text for result in results], example.gold_spans)
        rows.append(
            {
                "task": TASK_RETRIEVAL,
                "example_id": example.id,
                "question": example.question,
                "k": k,
                "retrieved_chunk_ids": [result.chunk_id for result in results],
                "first_hit_rank": -1 if hit is None else hit,
                "recall_at_k": recall_at_k(hit, k),
                "reciprocal_rank": reciprocal_rank(hit),
                "ndcg_at_10": ndcg_at_k(hit, NDCG_K),
                "retrieval_latency_s": elapsed,
                "latency_s": elapsed,
                "difficulty": example.difficulty.value,
            }
        )
    return rows


def _answer_rows(
    stack: RetrievalStack,
    generator: Generator,
    judge: Judge | None,
    examples: list[AnswerExample],
    k: int,
    generator_cfg: GeneratorConfig,
    judge_cfg: JudgeConfig,
    pricing: PricingConfig,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        results, retrieval_latency = timed_retrieve(stack, example.question, k, repeats)

        generated = generator.generate(
            GenerationInput(example_id=example.id, question=example.question, context=results)
        )
        answer_tokens = generated.output_tokens or stack.tokenizer.count(generated.answer)
        measured_cost = generated.cost_usd(
            generator_cfg.input_usd_per_mtok, generator_cfg.output_usd_per_mtok
        )
        # Priced at the reference model's rates, not the generator's: for a
        # generator that makes no API call the generator's own rates are zero,
        # and "free" is not the answer to "what would this configuration cost".
        projected_cost = (
            generated.context_tokens * pricing.input_usd_per_mtok
            + answer_tokens * pricing.output_usd_per_mtok
        ) / USD_PER_MTOK

        row: dict[str, Any] = {
            "task": TASK_ANSWER,
            "example_id": example.id,
            "question": example.question,
            "k": k,
            "retrieved_chunk_ids": [result.chunk_id for result in results],
            "answer": generated.answer,
            "answer_hash": sha256_text(generated.answer),
            "generator_model": generated.model,
            "input_tokens": generated.input_tokens,
            "output_tokens": generated.output_tokens,
            "context_tokens": generated.context_tokens,
            "answer_tokens": answer_tokens,
            "cost_usd": measured_cost,
            "projected_cost_usd": projected_cost,
            "retrieval_latency_s": retrieval_latency,
            "generation_latency_s": generated.latency_s,
            "latency_s": retrieval_latency + generated.latency_s,
            "replayed": generated.replayed,
        }
        hit = first_hit_rank([result.text for result in results], example.gold_spans)
        row["recall_at_k"] = recall_at_k(hit, k) if example.gold_spans else None

        if judge is not None:
            score: JudgeScore = judge.score(
                JudgeInput(
                    example_id=example.id,
                    question=example.question,
                    answer=generated.answer,
                    context=results,
                    gold_spans=example.gold_spans,
                    generator=generated.model,
                )
            )
            row.update({f"judge_{axis}": value for axis, value in score.scores.as_dict().items()})
            row["judge_model"] = score.judge_model
            row["judge_latency_s"] = score.latency_s
            row["judge_input_tokens"] = score.input_tokens
            row["judge_output_tokens"] = score.output_tokens
            row["judge_cost_usd"] = (
                score.input_tokens * judge_cfg.input_usd_per_mtok
                + score.output_tokens * judge_cfg.output_usd_per_mtok
            ) / USD_PER_MTOK
        rows.append(row)
    return rows


def _aggregate(
    rows: pd.DataFrame, gate: GateConfig, judge_cfg: JudgeConfig, k: int
) -> dict[str, Any]:
    retrieval = rows[rows["task"] == TASK_RETRIEVAL]
    answers = rows[rows["task"] == TASK_ANSWER]

    metrics: dict[str, Any] = {
        "k": k,
        "n_retrieval": len(retrieval),
        "n_answers": len(answers),
        "recall_at_k": mean(retrieval["recall_at_k"].tolist()),
        "mrr": mean(retrieval["reciprocal_rank"].tolist()),
        "ndcg_at_10": mean(retrieval["ndcg_at_10"].tolist()),
    }

    for axis in judge_cfg.axes:
        column = f"judge_{axis}"
        if column in answers:
            metrics[f"judge_{axis}"] = mean(answers[column].tolist())
            metrics[f"judge_{axis}_normalised"] = normalise_score(
                metrics[f"judge_{axis}"], judge_cfg.scale_min, judge_cfg.scale_max
            )

    latencies = rows["latency_s"].tolist()
    metrics["p50_latency_s"] = percentile(latencies, 50)
    metrics["p95_latency_s"] = percentile(latencies, 95)
    metrics["mean_latency_s"] = mean(latencies)

    if len(answers):
        metrics["cost_per_query_usd"] = mean(answers["cost_usd"].tolist())
        metrics["projected_cost_per_query_usd"] = mean(answers["projected_cost_usd"].tolist())
        metrics["input_tokens_per_query"] = mean(answers["input_tokens"].tolist())
        metrics["output_tokens_per_query"] = mean(answers["output_tokens"].tolist())
        metrics["context_tokens_per_query"] = mean(answers["context_tokens"].tolist())
        if "judge_cost_usd" in answers:
            metrics["eval_cost_usd"] = float(answers["judge_cost_usd"].sum())
    else:
        metrics["cost_per_query_usd"] = 0.0
        metrics["projected_cost_per_query_usd"] = 0.0

    components = {
        name.removeprefix("judge_").removesuffix("_normalised"): value
        for name, value in metrics.items()
        if name.endswith("_normalised")
    }
    components["recall_at_k"] = metrics["recall_at_k"]
    metrics["composite_quality"] = composite_quality(components, dict(gate.composite_weights))
    metrics["composite_components"] = components
    return metrics


@dataclass(frozen=True, slots=True)
class EvalSets:
    """The examples a run scores, already bound to its corpus."""

    retrieval: list[RetrievalExample]
    answers: list[AnswerExample]

    @property
    def hashes(self) -> dict[str, str]:
        """Content hashes, recorded in the run so a changed question is visible."""
        return {
            "retrieval": hash_obj([example.model_dump(mode="json") for example in self.retrieval]),
            "answers": hash_obj([example.model_dump(mode="json") for example in self.answers]),
        }


def load_eval_sets(cfg: DictConfig, corpus: LoadedCorpus) -> EvalSets:
    """Read the accepted examples and refuse any the corpus cannot answer."""
    retrieval_path = resolve_path(cfg, "evalsets.retrieval_path")
    answers_path = resolve_path(cfg, "evalsets.answers_path")
    retrieval_examples = [
        example
        for example in read_jsonl(retrieval_path, RetrievalExample)
        if example.status is ReviewStatus.ACCEPTED
    ]
    answer_examples = read_jsonl(answers_path, AnswerExample)
    limit = cfg.evalsets.get("limit")
    if limit:
        retrieval_examples = retrieval_examples[: int(limit)]
        answer_examples = answer_examples[: int(limit)]
    check_bound(corpus, retrieval_examples, answer_examples, retrieval_path, answers_path)
    return EvalSets(retrieval=retrieval_examples, answers=answer_examples)


def measured_fingerprint(
    stack: RetrievalStack, generator: Generator, judge: Judge | None
) -> dict[str, Any]:
    """The identity of every component a run measures."""
    return {
        "chunker": stack.chunker.fingerprint(),
        "embedder": stack.embedder.fingerprint(),
        "retriever": stack.retriever.fingerprint(),
        "generator": generator.fingerprint(),
        "judge": judge.fingerprint() if judge else None,
    }


def evaluate(
    cfg: DictConfig,
    stack: RetrievalStack,
    generator: Generator,
    judge: Judge | None,
    prompts: dict[str, Prompt],
) -> RunOutcome:
    """Run both eval sets against one configuration."""
    generator_cfg = typed_node(cfg, "generator", GeneratorConfig)
    judge_cfg = typed_node(cfg, "judge", JudgeConfig)
    gate = typed_node(cfg, "gate", GateConfig)
    pricing = typed_node(cfg, "pricing", PricingConfig)
    k = int(cfg.retriever.k)

    timing = typed_node(cfg, "timing", TimingConfig)

    eval_sets = load_eval_sets(cfg, stack.corpus)
    if timing.warmup:
        # Untimed: the first queries pay for lazy loads and cold caches, and
        # that cost belongs to process start-up, not to any one question.
        questions = [example.question for example in eval_sets.retrieval]
        questions += [example.question for example in eval_sets.answers]
        for question in questions:
            stack.retriever.retrieve(question, k=k)
    rows = pd.DataFrame(
        _retrieval_rows(stack, eval_sets.retrieval, k, timing.repeats)
        + _answer_rows(
            stack,
            generator,
            judge,
            eval_sets.answers,
            k,
            generator_cfg,
            judge_cfg,
            pricing,
            timing.repeats,
        )
    )
    metrics = _aggregate(rows, gate, judge_cfg, k)

    digest = config_hash(cfg)
    meta = RunMeta(
        run_id=make_run_id(digest),
        created_at=utc_now_iso(),
        config_hash=digest,
        corpus_name=stack.corpus.name,
        corpus_hash=stack.corpus.corpus_hash,
        index_hash=stack.index.meta.index_hash,
        prompt_hashes={name: prompt.sha256 for name, prompt in prompts.items()},
        evalset_hashes=eval_sets.hashes,
        api_mode=str(cfg.api.mode),
        replayed=bool(rows.get("replayed", pd.Series(dtype=bool)).any()),
        fingerprint=measured_fingerprint(stack, generator, judge),
        timing={"warmup": timing.warmup, "repeats": timing.repeats, "statistic": "median"},
    )
    return RunOutcome(meta=meta, rows=rows, metrics=metrics)
