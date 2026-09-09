from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from evalgate.models.base import ModelRequest, ModelResponse, SchemaViolationError, ToolSpec
from evalgate.models.cache import ResponseCache
from evalgate.models.cassette import CassetteMissError, CassetteStore
from evalgate.models.client import CachingClient, RecordingClient, ReplayClient
from evalgate.models.schema import tool_schema
from evalgate.models.structured import build_tool, call_structured
from tests.fakes import ScriptedClient, response


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int
    reason: str


class Nested(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[Verdict]


def request(prompt: str = "hello", model: str = "test-model") -> ModelRequest:
    return ModelRequest(
        model=model,
        prompt=prompt,
        max_tokens=64,
        temperature=0.0,
        tool=build_tool("submit", "submit a verdict", Verdict),
        purpose="judge",
        prompt_hash="a" * 64,
    )


def test_cache_key_is_stable() -> None:
    assert request().cache_key() == request().cache_key()


def test_cache_key_changes_with_the_prompt() -> None:
    assert replace(request(), prompt="different").cache_key() != request().cache_key()


def test_cache_key_changes_with_the_model() -> None:
    assert replace(request(), model="other").cache_key() != request().cache_key()


def test_cache_key_changes_with_max_tokens() -> None:
    assert replace(request(), max_tokens=65).cache_key() != request().cache_key()


def test_cache_key_changes_with_temperature() -> None:
    assert replace(request(), temperature=0.5).cache_key() != request().cache_key()


def test_cache_key_changes_with_the_prompt_hash() -> None:
    """Editing a prompt file must never be served from cache."""
    assert replace(request(), prompt_hash="b" * 64).cache_key() != request().cache_key()


def test_cache_key_ignores_purpose() -> None:
    """Two identical calls are the same call regardless of what asked for them."""
    base = request()
    assert replace(base, purpose="generation").cache_key() == base.cache_key()


def test_cost_is_computed_from_usage() -> None:
    reply = response(input_tokens=1_000_000, output_tokens=1_000_000)
    assert reply.cost_usd(3.0, 15.0) == pytest.approx(18.0)


def test_tool_schema_forbids_extras_and_inlines_refs() -> None:
    schema = tool_schema(Nested)
    assert schema["additionalProperties"] is False
    assert "$defs" not in schema
    item = schema["properties"]["verdicts"]["items"]
    assert "$ref" not in item
    assert set(item["properties"]) == {"score", "reason"}


def test_tool_spec_wire_form_is_strict() -> None:
    spec: ToolSpec = build_tool("submit", "d", Verdict)
    assert spec.to_api()["strict"] is True


def test_cache_round_trip(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.sqlite")
    assert cache.get(request()) is None
    cache.put(request(), response({"score": 4, "reason": "ok"}))
    hit = cache.get(request())
    assert hit is not None
    assert hit.cache_hit is True
    assert hit.tool_input == {"score": 4, "reason": "ok"}
    assert cache.size() == 1


def test_cassette_round_trip_preserves_measured_latency(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path / "cassettes")
    store.save(request(), response({"score": 4, "reason": "ok"}, latency_s=1.25))
    replayed = store.load(request())
    assert replayed is not None
    assert replayed.replayed is True
    assert replayed.latency_s == 1.25


def test_replay_client_refuses_to_reach_the_network(tmp_path: Path) -> None:
    client = ReplayClient(store=CassetteStore(tmp_path))
    with pytest.raises(CassetteMissError, match="no cassette"):
        client.complete(request())


def test_replay_client_serves_a_recorded_call(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path)
    store.save(request(), response({"score": 2, "reason": "thin"}))
    assert ReplayClient(store=store).complete(request()).tool_input == {
        "score": 2,
        "reason": "thin",
    }


def test_caching_client_calls_through_once(tmp_path: Path) -> None:
    inner = ScriptedClient(responses=[response({"score": 5, "reason": "y"})])
    client = CachingClient(inner=inner, cache=ResponseCache(tmp_path / "c.sqlite"))
    first = client.complete(request())
    second = client.complete(request())
    assert len(inner.seen) == 1
    assert first.tool_input == second.tool_input
    assert second.cache_hit is True


def test_recording_client_writes_a_cassette_per_call(tmp_path: Path) -> None:
    store = CassetteStore(tmp_path / "cassettes")
    inner = ScriptedClient(responses=[response({"score": 5, "reason": "y"})])
    RecordingClient(inner=inner, store=store).complete(request())
    assert store.count() == 1


def test_call_structured_validates(repo_root: Path) -> None:
    client = ScriptedClient(responses=[response({"score": 4, "reason": "ok"})])
    result = call_structured(client, request(), Verdict, repo_root / "prompts", max_attempts=3)
    assert result.value.score == 4
    assert result.attempts == 1


def test_call_structured_repairs_then_succeeds(repo_root: Path) -> None:
    """A schema violation shows the model its own error rather than resending."""
    client = ScriptedClient(
        responses=[
            response({"score": "not a number", "reason": "x"}),
            response({"score": 3, "reason": "fixed"}),
        ]
    )
    result = call_structured(client, request(), Verdict, repo_root / "prompts", max_attempts=3)
    assert result.value.score == 3
    assert result.attempts == 2
    assert "did not satisfy" in client.seen[1].prompt
    assert client.seen[0].prompt != client.seen[1].prompt


def test_call_structured_fails_hard_rather_than_defaulting(repo_root: Path) -> None:
    """The single most dangerous thing this repo could do is invent a score."""
    client = ScriptedClient(responses=[response({"score": "bad", "reason": "x"})] * 3)
    with pytest.raises(SchemaViolationError, match="failed to produce valid Verdict in 3"):
        call_structured(client, request(), Verdict, repo_root / "prompts", max_attempts=3)


def test_call_structured_fails_when_no_tool_was_used(repo_root: Path) -> None:
    client = ScriptedClient(responses=[response(None, text="I would rather explain in prose")] * 2)
    with pytest.raises(SchemaViolationError, match="no tool_use block"):
        call_structured(client, request(), Verdict, repo_root / "prompts", max_attempts=2)


def test_response_defaults_are_not_scores() -> None:
    """ModelResponse must not carry a plausible-looking default judgment."""
    blank: ModelResponse = response(None)
    assert blank.tool_input is None
