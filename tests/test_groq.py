from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict

from evalgate.config import load_config
from evalgate.generation.factory import build_generator
from evalgate.hashing import hash_obj
from evalgate.ingest.tokens import RegexTokenEstimator
from evalgate.judging.factory import build_judge
from evalgate.models.base import (
    ModelClient,
    ModelError,
    ModelRequest,
    ModelResponse,
)
from evalgate.models.client import build_model_client
from evalgate.models.groq_client import (
    GroqClient,
    MissingCredentialsError,
    build_payload,
    parse_response,
    tool_to_openai,
)
from evalgate.models.structured import build_tool, call_structured

TOKENIZER = RegexTokenEstimator(name="t", pattern=r"\w+|[^\w\s]", tokens_per_word=1.3)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int
    reason: str


def request(system: str | None = None, with_tool: bool = True) -> ModelRequest:
    return ModelRequest(
        model="llama-3.3-70b-versatile",
        prompt="grade this",
        max_tokens=256,
        temperature=0.0,
        system=system,
        tool=build_tool("submit", "submit a verdict", Verdict) if with_tool else None,
        purpose="judge",
        prompt_hash="a" * 64,
    )


def completion(arguments: object, content: str = "") -> dict[str, object]:
    """A chat completion shaped the way OpenAI-compatible APIs return one."""
    return {
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": content,
                    "tool_calls": [{"function": {"name": "submit", "arguments": arguments}}],
                },
            }
        ],
        "usage": {"prompt_tokens": 900, "completion_tokens": 40},
    }


def test_tool_is_translated_into_the_function_shape() -> None:
    translated = tool_to_openai(build_tool("submit", "submit a verdict", Verdict))
    assert translated["type"] == "function"
    assert translated["function"]["name"] == "submit"
    # The schema moves from input_schema to parameters, unchanged.
    assert translated["function"]["parameters"]["additionalProperties"] is False
    assert set(translated["function"]["parameters"]["properties"]) == {"score", "reason"}


def test_the_tool_is_forced() -> None:
    """A score arrives through a validated call or not at all."""
    payload = build_payload(request())
    assert payload["tool_choice"] == {"type": "function", "function": {"name": "submit"}}
    assert len(payload["tools"]) == 1


def test_payload_carries_the_sampling_parameters() -> None:
    payload = build_payload(request())
    assert payload["model"] == "llama-3.3-70b-versatile"
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 0.0
    assert payload["messages"] == [{"role": "user", "content": "grade this"}]


def test_a_system_prompt_becomes_a_system_message() -> None:
    payload = build_payload(request(system="you are a judge"))
    assert payload["messages"][0] == {"role": "system", "content": "you are a judge"}
    assert payload["messages"][1]["role"] == "user"


def test_no_tool_means_no_tool_choice() -> None:
    payload = build_payload(request(with_tool=False))
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_tool_arguments_arrive_as_a_string_and_are_decoded() -> None:
    """OpenAI returns arguments as JSON text; Anthropic returns them parsed.

    Forgetting to decode hands the judge a string where a mapping belongs, and
    the schema retry loop burns every attempt before failing.
    """
    response = parse_response(completion(json.dumps({"score": 4, "reason": "ok"})), 1.5, 1)
    assert response.tool_input == {"score": 4, "reason": "ok"}


def test_already_parsed_arguments_are_accepted() -> None:
    response = parse_response(completion({"score": 4, "reason": "ok"}), 1.0, 1)
    assert response.tool_input == {"score": 4, "reason": "ok"}


def test_undecodable_arguments_become_no_tool_input_not_a_guess() -> None:
    assert parse_response(completion("{not json"), 1.0, 1).tool_input is None


def test_a_json_scalar_is_not_accepted_as_tool_input() -> None:
    assert parse_response(completion("42"), 1.0, 1).tool_input is None


def test_usage_and_latency_are_recorded() -> None:
    response = parse_response(completion(json.dumps({"score": 1, "reason": "x"})), 2.25, 3)
    assert response.input_tokens == 900
    assert response.output_tokens == 40
    assert response.latency_s == 2.25
    assert response.attempts == 3
    assert response.stop_reason == "tool_calls"


def test_a_response_with_no_tool_call_yields_text_only() -> None:
    body = {
        "model": "m",
        "choices": [{"finish_reason": "stop", "message": {"content": "  prose  "}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    response = parse_response(body, 1.0, 1)
    assert response.tool_input is None
    assert response.text == "prose"


def test_an_empty_response_does_not_raise() -> None:
    assert parse_response({}, 1.0, 1).tool_input is None


def test_a_missing_key_names_the_variable_and_where_to_get_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = GroqClient(timeout_s=1.0, max_retries=1, backoff_initial_s=0.0, backoff_max_s=0.0)
    with pytest.raises(MissingCredentialsError, match="GROQ_API_KEY"):
        client.complete(request())


def test_retry_after_is_honoured_over_exponential_backoff() -> None:
    """Rate limits are the binding constraint on a free tier."""
    import httpx

    client = GroqClient(timeout_s=1.0, max_retries=3, backoff_initial_s=8.0, backoff_max_s=60.0)
    response = httpx.Response(429, headers={"retry-after": "2"})
    assert client._backoff(1, response) == 2.0
    assert client._backoff(1, None) >= 8.0


def test_retry_after_is_capped() -> None:
    import httpx

    client = GroqClient(timeout_s=1.0, max_retries=3, backoff_initial_s=1.0, backoff_max_s=5.0)
    assert client._backoff(1, httpx.Response(429, headers={"retry-after": "900"})) == 5.0


def vendor_of(client: ModelClient) -> str:
    """The vendor at the bottom of the stack, under the cache and the recorder.

    The wrappers are the point of the stack, so the test peels them rather than
    bypassing `build_model_client` and testing a dispatch nothing calls.
    """
    while (inner := getattr(client, "inner", None)) is not None:
        client = inner
    return client.name


def test_the_client_stack_dispatches_on_the_configured_vendor() -> None:
    cfg: DictConfig = load_config(overrides=["+experiment=groq"])
    assert vendor_of(build_model_client(cfg)) == "groq"


def test_anthropic_remains_the_default_vendor() -> None:
    cfg: DictConfig = load_config(overrides=["api=live"])
    assert vendor_of(build_model_client(cfg)) == "anthropic"


def test_the_cache_wraps_the_vendor_rather_than_replacing_it() -> None:
    """Peeling must not be able to pass by finding nothing to peel."""
    cfg: DictConfig = load_config(overrides=["+experiment=groq"])
    assert build_model_client(cfg).name == "cached"


def test_an_unknown_vendor_is_refused() -> None:
    cfg: DictConfig = load_config(overrides=["api=live", "api.provider=nope"])
    with pytest.raises(ValueError, match=re.escape("api.provider must be one of")):
        build_model_client(cfg)


def test_the_groq_experiment_wires_the_whole_stack() -> None:
    """One flag, so the vendor, the mode and both models cannot drift apart."""
    cfg: DictConfig = load_config(overrides=["+experiment=groq"])
    assert cfg.api.provider == "groq"
    assert cfg.api.mode == "live"
    assert cfg.judge.provider == "api"
    assert cfg.generator.provider == "api"
    assert cfg.embedder.name == "local"


def test_the_judge_does_not_grade_its_own_model_by_default() -> None:
    """A judge grading its own output confounds the self-preference probe.

    The probe reports a gap between two generators. If one of them is the
    judge, the gap is evidence about self-preference; if neither is, the gap
    is uninterpretable, and the config would have quietly made it so.
    """
    cfg: DictConfig = load_config(overrides=["+experiment=groq"])
    assert cfg.judge.model != cfg.generator.model
    contrast: DictConfig = load_config(
        overrides=["+experiment=groq", f"generator={cfg.probes.contrast_generator}"]
    )
    assert contrast.generator.model == cfg.judge.model


def test_the_judge_and_generator_build_on_groq() -> None:
    cfg: DictConfig = load_config(overrides=["+experiment=groq"])
    client = build_model_client(cfg)
    assert build_judge(cfg, client).model == cfg.judge.model
    assert build_generator(cfg, TOKENIZER, client).model == cfg.generator.model


def test_the_original_anthropic_spelling_still_works() -> None:
    """Existing configs and committed run records must keep loading."""
    cfg: DictConfig = load_config(overrides=["+experiment=live"])
    assert cfg.judge.provider == "anthropic"
    assert build_judge(cfg, build_model_client(cfg)).model == cfg.judge.model


def test_an_unknown_judge_provider_is_refused() -> None:
    cfg: DictConfig = load_config(overrides=["judge.provider=nonsense"])
    with pytest.raises(ValueError, match="unknown judge provider"):
        build_judge(cfg, None)


def test_reasoning_effort_is_sent_only_when_set() -> None:
    """gpt-oss will not emit a forced tool call without it; Groq 400s instead."""
    assert "reasoning_effort" not in build_payload(request())
    payload = build_payload(
        ModelRequest(
            model="openai/gpt-oss-120b",
            prompt="grade this",
            max_tokens=4096,
            temperature=0.0,
            reasoning_effort="low",
            tool=build_tool("submit", "submit a verdict", Verdict),
            purpose="judge",
            prompt_hash="a" * 64,
        )
    )
    assert payload["reasoning_effort"] == "low"


def test_reasoning_effort_changes_the_cache_key() -> None:
    """It changes the response, so it must not be served from the same entry."""
    base = request()
    assert replace(base, reasoning_effort="low").cache_key() != base.cache_key()
    assert (
        replace(base, reasoning_effort="low").cache_key()
        != replace(base, reasoning_effort="high").cache_key()
    )


def test_an_unset_reasoning_effort_leaves_existing_cache_keys_alone() -> None:
    """Every cassette committed before this field existed was keyed without it.

    Including the field unconditionally would miss all of them and turn replay
    CI red on a change that alters no request, so the key must be byte-identical
    to what it was when reasoning_effort is None.
    """
    assert request().cache_key() == hash_obj(
        {
            "model": "llama-3.3-70b-versatile",
            "prompt": "grade this",
            "system": None,
            "max_tokens": 256,
            "temperature": 0.0,
            "tool": build_tool("submit", "submit a verdict", Verdict).to_api(),
            "prompt_hash": "a" * 64,
        }
    )


def test_the_retry_loop_carries_reasoning_effort() -> None:
    """A retry that drops it retries a different call than the one that failed."""
    attempted: list[ModelRequest] = []

    class Recorder:
        name = "recorder"

        def complete(self, req: ModelRequest) -> ModelResponse:
            attempted.append(req)
            return ModelResponse(
                model=req.model,
                text="",
                input_tokens=1,
                output_tokens=1,
                latency_s=0.0,
                tool_input={"score": 4, "reason": "ok"},
            )

    original = replace(request(), reasoning_effort="low")
    call_structured(Recorder(), original, Verdict, Path("prompts"), max_attempts=2)
    assert attempted and all(item.reasoning_effort == "low" for item in attempted)


def test_anthropic_refuses_reasoning_effort_rather_than_dropping_it() -> None:
    """A sampling parameter that silently does nothing fakes comparability."""
    from evalgate.models.anthropic_client import AnthropicClient

    client = AnthropicClient(timeout_s=1.0, max_retries=1, backoff_initial_s=0.0, backoff_max_s=0.0)
    with pytest.raises(ModelError, match="reasoning_effort is not supported"):
        client.complete(replace(request(), reasoning_effort="low"))


def test_the_groq_judge_needs_no_reasoning_effort_to_call_its_tool() -> None:
    """The default judge is the one that complies without being suppressed."""
    cfg: DictConfig = load_config(overrides=["+experiment=groq"])
    assert cfg.judge.reasoning_effort is None
    alternative: DictConfig = load_config(overrides=["+experiment=groq", "judge=groq_gptoss"])
    assert alternative.judge.reasoning_effort == "low"
