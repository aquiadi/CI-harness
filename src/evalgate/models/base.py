"""Request and response shapes for model calls.

Everything that reaches a model goes through ``ModelRequest``, and everything
that comes back is a ``ModelResponse`` carrying its own usage, cost and
latency. That is what makes the cost and latency columns in the Pareto table
measurements rather than estimates: they are read off the responses that
actually produced the answers being scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from evalgate.errors import EvalgateError
from evalgate.hashing import hash_obj

USD_PER_MTOK_DIVISOR = 1_000_000.0


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool definition used to constrain the model's output shape."""

    name: str
    description: str
    input_schema: dict[str, Any]
    strict: bool = True

    def to_api(self) -> dict[str, Any]:
        """The wire form the Messages API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "strict": self.strict,
        }


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One call to a model, in a form that can be hashed and replayed."""

    model: str
    prompt: str
    max_tokens: int
    temperature: float
    system: str | None = None
    tool: ToolSpec | None = None
    # Identifies what the call was for (generation, judging, eval generation).
    purpose: str = "generation"
    # The hash of the prompt file this call was rendered from. Part of the
    # cache key, so editing a prompt cannot serve a stale cached answer.
    prompt_hash: str = ""

    def cache_key(self) -> str:
        """Key on everything that can change the response.

        Deliberately includes the prompt hash and excludes ``purpose``: two
        calls with identical inputs to the same model are the same call, and
        a prompt edit must never be served from cache.
        """
        return hash_obj(
            {
                "model": self.model,
                "prompt": self.prompt,
                "system": self.system,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "tool": self.tool.to_api() if self.tool else None,
                "prompt_hash": self.prompt_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """What a model returned, with what it cost to get it."""

    model: str
    text: str
    tool_input: dict[str, Any] | None
    input_tokens: int
    output_tokens: int
    latency_s: float
    stop_reason: str | None = None
    cache_hit: bool = False
    replayed: bool = False
    attempts: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def cost_usd(self, input_usd_per_mtok: float, output_usd_per_mtok: float) -> float:
        """Cost of this call at the configured prices."""
        return (
            self.input_tokens * input_usd_per_mtok + self.output_tokens * output_usd_per_mtok
        ) / USD_PER_MTOK_DIVISOR


class ModelError(EvalgateError):
    """Raised when a model call cannot be completed."""


class SchemaViolationError(ModelError):
    """Raised when the model's tool input does not satisfy the schema.

    Never caught and turned into a default score. A judgment that could not be
    parsed is missing data, and missing data must be visible.
    """


@runtime_checkable
class ModelClient(Protocol):
    """Anything that can answer a ModelRequest."""

    name: str

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute one request."""
        ...
