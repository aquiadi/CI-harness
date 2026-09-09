from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from evalgate.config import load_config
from evalgate.generation.base import GenerationInput
from evalgate.generation.extractive import NO_ANSWER, ExtractiveGenerator
from evalgate.generation.factory import build_generator
from evalgate.generation.llm import LLMGenerator
from evalgate.ingest.tokens import RegexTokenEstimator
from evalgate.prompts import load_prompt
from evalgate.retrieval.base import Retrieved
from tests.fakes import ScriptedClient, response

TOKENIZER = RegexTokenEstimator(name="t", pattern=r"\w+|[^\w\s]", tokens_per_word=1.3)


def chunk(chunk_id: str, text: str, rank: int = 0) -> Retrieved:
    return Retrieved(
        chunk_id=chunk_id,
        score=1.0 - rank * 0.1,
        rank=rank,
        text=text,
        doc_id=chunk_id.split("#")[0],
        doc_title="Doc",
    )


def extractive() -> ExtractiveGenerator:
    return ExtractiveGenerator(
        name="extractive",
        model="extractive-v1",
        tokenizer=TOKENIZER,
        max_sentences=2,
        max_chunks=3,
        min_overlap=2,
    )


def item(question: str, context: list[Retrieved]) -> GenerationInput:
    return GenerationInput(example_id="e-1", question=question, context=context)


def test_extractive_quotes_the_matching_sentence_with_its_citation() -> None:
    context = [
        chunk("d#0001", "Reports are due one month after the quarter ends. Unrelated sentence."),
    ]
    generated = extractive().generate(item("When are reports due?", context))
    assert "one month after the quarter ends" in generated.answer
    assert "[d#0001]" in generated.answer


def test_extractive_declines_when_nothing_overlaps() -> None:
    context = [chunk("d#0001", "Aluminium smelting uses prebaked anodes.")]
    assert extractive().generate(item("What is the VAT rate?", context)).answer == NO_ANSWER


def test_extractive_is_deterministic() -> None:
    context = [chunk("d#0001", "Reports are due one month after the quarter ends.")]
    first = extractive().generate(item("When are reports due?", context)).answer
    second = extractive().generate(item("When are reports due?", context)).answer
    assert first == second


def test_extractive_respects_its_sentence_budget() -> None:
    text = " ".join(f"Reports are due sentence {index}." for index in range(6))
    generated = extractive().generate(item("When are reports due?", [chunk("d#0001", text)]))
    assert generated.answer.count("[d#0001]") == 2


def test_extractive_costs_nothing_and_reports_zero_api_tokens() -> None:
    context = [chunk("d#0001", "Reports are due one month after the quarter ends.")]
    generated = extractive().generate(item("When are reports due?", context))
    assert generated.input_tokens == 0
    assert generated.output_tokens == 0
    assert generated.cost_usd(3.0, 15.0) == 0.0
    assert generated.context_tokens > 0


def test_llm_generator_records_usage_and_latency(repo_root: Path) -> None:
    client = ScriptedClient(
        responses=[response(text="An answer [d#0001].", input_tokens=900, output_tokens=40)]
    )
    generator = LLMGenerator(
        name="anthropic",
        model="test-model",
        client=client,
        prompt=load_prompt(repo_root / "prompts", "generation/answer_v1.md"),
        tokenizer=TOKENIZER,
        max_tokens=512,
        temperature=0.0,
    )
    generated = generator.generate(item("q?", [chunk("d#0001", "text")]))
    assert generated.answer == "An answer [d#0001]."
    assert generated.input_tokens == 900
    assert generated.cost_usd(3.0, 15.0) == pytest.approx((900 * 3.0 + 40 * 15.0) / 1e6)


def test_llm_generator_sends_the_context_and_the_question(repo_root: Path) -> None:
    client = ScriptedClient(responses=[response(text="answer")])
    generator = LLMGenerator(
        name="anthropic",
        model="test-model",
        client=client,
        prompt=load_prompt(repo_root / "prompts", "generation/answer_v1.md"),
        tokenizer=TOKENIZER,
        max_tokens=512,
        temperature=0.0,
    )
    generator.generate(item("When are reports due?", [chunk("d#0001", "the chunk body")]))
    sent = client.seen[0].prompt
    assert "the chunk body" in sent
    assert "When are reports due?" in sent
    assert client.seen[0].prompt_hash


def test_factory_builds_the_extractive_generator_without_a_client() -> None:
    cfg: DictConfig = load_config(overrides=["generator=extractive"])
    assert build_generator(cfg, TOKENIZER).model == "extractive-v1"


def test_factory_refuses_an_llm_generator_with_no_client() -> None:
    cfg: DictConfig = load_config(overrides=["generator=sonnet"])
    with pytest.raises(ValueError, match="needs a model client"):
        build_generator(cfg, TOKENIZER)
