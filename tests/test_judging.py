from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig
from pydantic import ValidationError

from evalgate.citations import audit_citations, extract_citations
from evalgate.config import load_config
from evalgate.context import context_chunk_ids, format_context
from evalgate.judging.base import JudgeInput
from evalgate.judging.factory import build_judge
from evalgate.judging.heuristic import HeuristicJudge
from evalgate.judging.llm import LLMJudge
from evalgate.judging.schema import build_judgment_model, split_judgment
from evalgate.models.base import SchemaViolationError
from evalgate.prompts import load_prompt
from evalgate.retrieval.base import Retrieved
from tests.fakes import ScriptedClient, response

AXES = ["groundedness", "relevance", "citation_correctness"]


def chunk(chunk_id: str, text: str, rank: int = 0) -> Retrieved:
    return Retrieved(
        chunk_id=chunk_id,
        score=1.0 - rank * 0.1,
        rank=rank,
        text=text,
        doc_id=chunk_id.split("#")[0],
        doc_title="Doc",
    )


def item(answer: str, context: list[Retrieved] | None = None) -> JudgeInput:
    return JudgeInput(
        example_id="e-1",
        question="When is the report due?",
        answer=answer,
        context=context
        or [chunk("d#0001", "The report is due one month after the end of the quarter.")],
        gold_spans=["due one month after the end of the quarter"],
    )


def judgment_payload(values: tuple[int, int, int] = (5, 4, 3)) -> dict[str, object]:
    payload: dict[str, object] = {}
    for axis, value in zip(AXES, values, strict=True):
        payload[axis] = value
        payload[f"{axis}_rationale"] = f"because of {axis}"
    return payload


def test_judgment_model_enforces_the_configured_scale() -> None:
    model = build_judgment_model(AXES, 1, 5)
    with pytest.raises(ValidationError):
        model.model_validate(judgment_payload((6, 4, 3)))


def test_judgment_model_requires_a_rationale_per_axis() -> None:
    model = build_judgment_model(AXES, 1, 5)
    payload = judgment_payload()
    payload["groundedness_rationale"] = ""
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_judgment_model_follows_the_configured_axes() -> None:
    model = build_judgment_model(["accuracy"], 0, 3)
    assert set(model.model_fields) == {"accuracy", "accuracy_rationale"}
    validated = model.model_validate({"accuracy": 3, "accuracy_rationale": "x"})
    assert validated.model_dump()["accuracy"] == 3


def test_judgment_model_rejects_an_empty_or_inverted_scale() -> None:
    with pytest.raises(ValueError, match="at least one axis"):
        build_judgment_model([], 1, 5)
    with pytest.raises(ValueError, match="must increase"):
        build_judgment_model(AXES, 5, 1)


def test_split_judgment_separates_scores_from_rationales() -> None:
    scores, rationales = split_judgment(judgment_payload(), AXES)
    assert scores == {"groundedness": 5, "relevance": 4, "citation_correctness": 3}
    assert rationales["relevance"] == "because of relevance"


def test_llm_judge_scores_through_a_validated_tool_call(repo_root: Path) -> None:
    client = ScriptedClient(responses=[response(judgment_payload())])
    judge = LLMJudge(
        name="anthropic",
        model="test-model",
        client=client,
        prompt=load_prompt(repo_root / "prompts", "judge/rubric_v1.md"),
        prompts_dir=repo_root / "prompts",
        axes=AXES,
        scale_min=1,
        scale_max=5,
        max_tokens=256,
        temperature=0.0,
        max_attempts=3,
    )
    score = judge.score(item("An answer [d#0001]."))
    assert score.scores.as_dict() == {
        "groundedness": 5,
        "relevance": 4,
        "citation_correctness": 3,
    }
    assert score.rationales["groundedness"]
    assert client.seen[0].tool is not None
    assert client.seen[0].tool.strict is True


def test_llm_judge_raises_rather_than_defaulting(repo_root: Path) -> None:
    """An unparseable judgment is missing data, never a middling score."""
    client = ScriptedClient(responses=[response({"groundedness": 9})] * 3)
    judge = LLMJudge(
        name="anthropic",
        model="test-model",
        client=client,
        prompt=load_prompt(repo_root / "prompts", "judge/rubric_v1.md"),
        prompts_dir=repo_root / "prompts",
        axes=AXES,
        scale_min=1,
        scale_max=5,
        max_tokens=256,
        temperature=0.0,
        max_attempts=3,
    )
    with pytest.raises(SchemaViolationError):
        judge.score(item("An answer."))


def heuristic() -> HeuristicJudge:
    return HeuristicJudge(
        name="heuristic", model="rule-based-v1", axes=AXES, scale_min=1, scale_max=5
    )


def test_heuristic_penalises_an_answer_the_context_does_not_support() -> None:
    supported = heuristic().score(
        item("The report is due one month after the end of the quarter [d#0001].")
    )
    invented = heuristic().score(
        item(
            "Reports are due within fourteen days and attract a surcharge of nine percent [d#0001]."
        )
    )
    assert supported.scores.groundedness > invented.scores.groundedness


def test_heuristic_penalises_a_citation_that_is_not_in_the_context() -> None:
    good = heuristic().score(item("The report is due one month later [d#0001]."))
    bad = heuristic().score(item("The report is due one month later [other#0009]."))
    assert good.scores.citation_correctness > bad.scores.citation_correctness


def test_heuristic_gives_the_floor_to_an_uncited_answer() -> None:
    assert (
        heuristic().score(item("The report is due one month later.")).scores.citation_correctness
        == 1
    )


def test_heuristic_rewards_a_refusal_only_when_the_evidence_is_absent() -> None:
    absent = JudgeInput(
        example_id="e",
        question="What is the VAT treatment?",
        answer="The context does not address VAT, so I cannot answer.",
        context=[chunk("d#0001", "Certificates are sold through a central platform.")],
        gold_spans=["nothing about VAT appears here"],
    )
    present = JudgeInput(
        example_id="e",
        question="When is the report due?",
        answer="The context does not address this, so I cannot answer.",
        context=[chunk("d#0001", "The report is due one month after the quarter.")],
        gold_spans=["The report is due one month after the quarter."],
    )
    assert heuristic().score(absent).scores.relevance == 5
    assert heuristic().score(present).scores.relevance == 1


def test_heuristic_is_deterministic() -> None:
    first = heuristic().score(item("An answer [d#0001]."))
    second = heuristic().score(item("An answer [d#0001]."))
    assert first.scores == second.scores


def test_heuristic_is_immune_to_context_order_by_construction() -> None:
    context = [chunk("d#0001", "alpha", 0), chunk("d#0002", "beta", 1)]
    forward = heuristic().score(item("alpha beta [d#0001]", context))
    reversed_ = heuristic().score(item("alpha beta [d#0001]", list(reversed(context))))
    assert forward.scores == reversed_.scores


def test_factory_builds_the_heuristic_judge_without_a_client() -> None:
    cfg: DictConfig = load_config(overrides=["judge=heuristic"])
    assert build_judge(cfg).model == "rule-based-v1"


def test_factory_refuses_an_llm_judge_with_no_client() -> None:
    cfg: DictConfig = load_config(overrides=["judge=sonnet"])
    with pytest.raises(ValueError, match="needs a model client"):
        build_judge(cfg)


def test_judge_is_swappable_to_a_different_model_than_the_generator() -> None:
    cfg: DictConfig = load_config(overrides=["judge=opus", "generator=sonnet"])
    assert cfg.judge.model != cfg.generator.model


def test_citations_are_extracted_in_order_without_duplicates() -> None:
    assert extract_citations("a [d#0001] b [d#0002] c [d#0001]") == ["d#0001", "d#0002"]


def test_citation_audit_splits_valid_from_dangling() -> None:
    audit = audit_citations("a [d#0001] b [x#0009]", ["d#0001", "d#0002"])
    assert audit.valid == ["d#0001"]
    assert audit.dangling == ["x#0009"]
    assert audit.precision == 0.5


def test_context_rendering_labels_every_chunk_with_its_id() -> None:
    rendered = format_context([chunk("d#0001", "alpha"), chunk("d#0002", "beta")])
    assert "[d#0001]" in rendered
    assert "[d#0002]" in rendered
    assert context_chunk_ids([chunk("d#0001", "alpha")]) == ["d#0001"]
