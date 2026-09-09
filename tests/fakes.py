"""Test doubles for model access."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalgate.models.base import ModelRequest, ModelResponse


@dataclass
class ScriptedClient:
    """Returns queued responses and records the requests it was given."""

    responses: list[ModelResponse]
    name: str = "scripted"
    seen: list[ModelRequest] = field(default_factory=list)

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Pop the next scripted response."""
        self.seen.append(request)
        if not self.responses:
            raise AssertionError("ScriptedClient ran out of responses")
        return self.responses.pop(0)


def response(
    tool_input: dict[str, Any] | None = None,
    text: str = "",
    model: str = "test-model",
    input_tokens: int = 100,
    output_tokens: int = 20,
    latency_s: float = 0.5,
) -> ModelResponse:
    """A response with sensible defaults."""
    return ModelResponse(
        model=model,
        text=text,
        tool_input=tool_input,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=latency_s,
        stop_reason="tool_use" if tool_input else "end_turn",
    )
