"""The Anthropic Messages API client.

Structured output is obtained by forcing a single tool and validating its input
against the pydantic model that produced the tool's schema. Nothing here parses
free text for scores.
"""

from __future__ import annotations

import os
import random
import time
from typing import TYPE_CHECKING, Any

from evalgate.models.base import ModelError, ModelRequest, ModelResponse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anthropic import Anthropic

API_KEY_ENV = "ANTHROPIC_API_KEY"


class MissingCredentialsError(ModelError):
    """Raised when a live call is attempted with no API credentials."""


class AnthropicClient:
    """Live Messages API access."""

    name = "anthropic"

    def __init__(
        self,
        timeout_s: float,
        max_retries: int,
        backoff_initial_s: float,
        backoff_max_s: float,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self._client: Anthropic | None = None

    def _sdk(self) -> Anthropic:
        if self._client is None:
            import anthropic

            if not os.environ.get(API_KEY_ENV):
                raise MissingCredentialsError(
                    f"{API_KEY_ENV} is not set. Use api=replay to run against committed "
                    "cassettes, or export a key to record new ones."
                )
            # SDK retries are disabled: this class owns the retry policy so that
            # the attempt count lands in the run record.
            self._client = anthropic.Anthropic(timeout=self.timeout_s, max_retries=0)
        return self._client

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            payload["system"] = request.system
        if request.tool is not None:
            payload["tools"] = [request.tool.to_api()]
            payload["tool_choice"] = {"type": "tool", "name": request.tool.name}
        return payload

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Call the API, retrying transient failures with exponential backoff."""
        import anthropic

        if request.reasoning_effort is not None:
            # Refused rather than dropped: a sampling parameter that silently
            # does nothing makes two runs look comparable when they are not.
            raise ModelError(
                "reasoning_effort is not supported by the Anthropic client "
                f"(got {request.reasoning_effort!r}); unset it or use api.provider=groq"
            )

        client = self._sdk()
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                message = client.messages.create(**self._payload(request))
            except (anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
                last = exc
            except anthropic.APIStatusError as exc:
                if exc.status_code < 500:
                    raise ModelError(f"{request.model}: {exc}") from exc
                last = exc
            else:
                elapsed = time.perf_counter() - started
                return _to_response(message, elapsed, attempt)

            if attempt < self.max_retries:
                time.sleep(self._backoff(attempt))
        raise ModelError(f"{request.model}: giving up after {self.max_retries} attempts: {last}")

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, so retries do not synchronise."""
        base: float = min(self.backoff_initial_s * float(2 ** (attempt - 1)), self.backoff_max_s)
        jitter: float = random.uniform(0.0, base * 0.25)
        return base + jitter


def _to_response(message: Any, elapsed: float, attempts: int) -> ModelResponse:
    """Extract text and tool input from a Messages API response."""
    text_parts: list[str] = []
    tool_input: dict[str, Any] | None = None
    for block in message.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use" and tool_input is None:
            # Tool input arrives as parsed JSON from the SDK; it is never
            # reconstructed from the serialised string.
            tool_input = dict(block.input)
    return ModelResponse(
        model=str(message.model),
        text="\n".join(text_parts).strip(),
        tool_input=tool_input,
        input_tokens=int(message.usage.input_tokens),
        output_tokens=int(message.usage.output_tokens),
        latency_s=elapsed,
        stop_reason=getattr(message, "stop_reason", None),
        attempts=attempts,
    )
