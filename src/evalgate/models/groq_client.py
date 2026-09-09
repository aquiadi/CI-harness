"""Groq, through its OpenAI-compatible chat completions API.

Groq exists here for one reason: it has a free tier, and the judge is the one
part of this repository that cannot run without a model. Everything the judge
needs -- a forced tool call whose arguments validate against a schema -- Groq
supports, so no part of the harness has to be weakened to accommodate it.

Two translations are all that separate it from the Anthropic client, and both
are the kind of detail that silently produces wrong data if you get them wrong:

*Tools.* Anthropic takes ``{name, description, input_schema}``; OpenAI takes
``{type: "function", function: {name, description, parameters}}``. The schema
itself is the same JSON Schema either way.

*Tool arguments.* Anthropic returns parsed JSON. OpenAI returns a **string**
that has to be decoded. A client that forgets to decode it hands the judge a
string where a mapping belongs, validation fails, and the retry loop burns
three attempts before failing -- so it is decoded here, and a payload that is
not a JSON object is reported as such rather than passed on.

Rate limits are the practical constraint on a free tier, so 429 is honoured
with the server's own ``retry-after`` when it sends one.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from evalgate.models.base import ModelError, ModelRequest, ModelResponse, ToolSpec

DEFAULT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
MODEL_HELP = (
    "Groq retires model ids regularly; check the current list at "
    "https://console.groq.com/docs/models and set judge.model / generator.model."
)


class MissingCredentialsError(ModelError):
    """Raised when a live call is attempted with no API credentials."""


def tool_to_openai(tool: ToolSpec) -> dict[str, Any]:
    """Translate a tool spec into OpenAI's function-calling shape."""
    function: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
    }
    if tool.strict:
        function["strict"] = True
    return {"type": "function", "function": function}


def build_payload(request: ModelRequest) -> dict[str, Any]:
    """The request body for one completion."""
    messages: list[dict[str, str]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    if request.reasoning_effort is not None:
        payload["reasoning_effort"] = request.reasoning_effort
    if request.tool is not None:
        payload["tools"] = [tool_to_openai(request.tool)]
        # Force the tool: the judge's contract is that a score arrives through
        # a validated call or not at all.
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": request.tool.name},
        }
    return payload


def _decode_arguments(raw: object) -> dict[str, Any] | None:
    """Decode OpenAI's tool arguments, which arrive as a JSON string."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def parse_response(body: dict[str, Any], elapsed: float, attempts: int) -> ModelResponse:
    """Turn a chat completion into the response shape the harness records."""
    choices = body.get("choices") or []
    message: dict[str, Any] = choices[0].get("message", {}) if choices else {}

    tool_input: dict[str, Any] | None = None
    for call in message.get("tool_calls") or []:
        tool_input = _decode_arguments(call.get("function", {}).get("arguments"))
        if tool_input is not None:
            break

    usage: dict[str, Any] = body.get("usage") or {}
    return ModelResponse(
        model=str(body.get("model", "")),
        text=str(message.get("content") or "").strip(),
        tool_input=tool_input,
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=int(usage.get("completion_tokens", 0)),
        latency_s=elapsed,
        stop_reason=str(choices[0].get("finish_reason")) if choices else None,
        attempts=attempts,
    )


@dataclass(slots=True)
class GroqClient:
    """Live access to Groq's chat completions API."""

    timeout_s: float
    max_retries: int
    backoff_initial_s: float
    backoff_max_s: float
    endpoint: str = DEFAULT_ENDPOINT
    api_key_env: str = "GROQ_API_KEY"
    name: str = "groq"

    def _key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise MissingCredentialsError(
                f"{self.api_key_env} is not set. Use api=replay to run against committed "
                "cassettes, or get a free key at https://console.groq.com/keys"
            )
        return key

    def _backoff(self, attempt: int, response: httpx.Response | None) -> float:
        """Server's retry-after when offered, else exponential with jitter."""
        if response is not None:
            header = response.headers.get("retry-after")
            if header:
                try:
                    return min(float(header), self.backoff_max_s)
                except ValueError:
                    pass
        base: float = min(self.backoff_initial_s * float(2 ** (attempt - 1)), self.backoff_max_s)
        jitter: float = random.uniform(0.0, base * 0.25)
        return base + jitter

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Call the API, retrying rate limits and server errors."""
        headers = {
            "Authorization": f"Bearer {self._key()}",
            "Content-Type": "application/json",
        }
        payload = build_payload(request)
        last = ""

        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            response: httpx.Response | None = None
            try:
                response = httpx.post(
                    self.endpoint, json=payload, headers=headers, timeout=self.timeout_s
                )
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if response.is_success:
                    return parse_response(response.json(), time.perf_counter() - started, attempt)
                last = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in RETRYABLE_STATUS:
                    hint = f"\n{MODEL_HELP}" if "model" in response.text.lower() else ""
                    raise ModelError(f"{request.model}: {last}{hint}")

            if attempt < self.max_retries:
                time.sleep(self._backoff(attempt, response))

        raise ModelError(f"{request.model}: giving up after {self.max_retries} attempts: {last}")
