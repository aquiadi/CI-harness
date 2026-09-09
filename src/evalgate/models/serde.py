"""Serialising model responses for the cache and for cassettes."""

from __future__ import annotations

from typing import Any

from evalgate.models.base import ModelResponse


def response_to_dict(response: ModelResponse) -> dict[str, Any]:
    """Plain-dict form, stable enough to diff in a pull request."""
    return {
        "model": response.model,
        "text": response.text,
        "tool_input": response.tool_input,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "latency_s": round(response.latency_s, 6),
        "stop_reason": response.stop_reason,
        "attempts": response.attempts,
        "extra": response.extra,
    }


def response_from_dict(payload: dict[str, Any], replayed: bool, cache_hit: bool) -> ModelResponse:
    """Rebuild a response, marking how it was obtained."""
    return ModelResponse(
        model=str(payload["model"]),
        text=str(payload["text"]),
        tool_input=payload.get("tool_input"),
        input_tokens=int(payload["input_tokens"]),
        output_tokens=int(payload["output_tokens"]),
        # The latency of the live call that produced this response is kept, not
        # the microseconds it took to read it back. A replayed run reports the
        # latency that was actually measured; `replayed` marks where it came
        # from so reports can say so.
        latency_s=float(payload["latency_s"]),
        stop_reason=payload.get("stop_reason"),
        cache_hit=cache_hit,
        replayed=replayed,
        attempts=int(payload.get("attempts", 1)),
        extra=dict(payload.get("extra") or {}),
    )
