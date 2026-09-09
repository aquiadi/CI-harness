"""Structured model output, validated against a pydantic model.

The contract is deliberately narrow: force one tool, validate its input, and if
validation fails, show the model its own error and ask again. After the
configured number of attempts the call raises.

It never falls back to parsing text, and it never substitutes a default. A
judgment that could not be validated is missing data. Missing data that
silently becomes a 3 is how an eval harness starts lying.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import BaseModel, ValidationError

from evalgate.models.base import (
    ModelClient,
    ModelRequest,
    ModelResponse,
    SchemaViolationError,
    ToolSpec,
)
from evalgate.prompts import Prompt, load_prompt

REPAIR_PROMPT = "system/schema_repair_v1.md"


@dataclass(frozen=True, slots=True)
class StructuredResult[T: BaseModel]:
    """A validated tool call and the response that carried it."""

    value: T
    response: ModelResponse
    attempts: int


def build_tool[T: BaseModel](name: str, description: str, model: type[T]) -> ToolSpec:
    """A strict tool whose input schema is the pydantic model."""
    from evalgate.models.schema import tool_schema

    return ToolSpec(
        name=name, description=description, input_schema=tool_schema(model), strict=True
    )


def call_structured[T: BaseModel](
    client: ModelClient,
    request: ModelRequest,
    model: type[T],
    prompts_dir: Path,
    max_attempts: int,
    backoff_initial_s: float = 0.0,
) -> StructuredResult[T]:
    """Call the model and validate its tool input, repairing on failure."""
    if request.tool is None:
        raise ValueError("call_structured requires a request with a tool")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    repair: Prompt = load_prompt(prompts_dir, REPAIR_PROMPT)
    prompt = request.prompt
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        # `replace` rather than a field-by-field rebuild: the retry must differ
        # from the original in the prompt and nothing else, and a rebuild
        # silently drops any field added to ModelRequest later.
        attempt_request = replace(request, prompt=prompt)
        response = client.complete(attempt_request)
        if response.tool_input is None:
            last_error = f"no tool_use block in the response (stop_reason={response.stop_reason})"
        else:
            try:
                return StructuredResult(
                    value=model.model_validate(response.tool_input),
                    response=response,
                    attempts=attempt,
                )
            except ValidationError as exc:
                last_error = str(exc)

        if attempt < max_attempts:
            if backoff_initial_s:
                time.sleep(backoff_initial_s * (2 ** (attempt - 1)))
            # Show the model the error rather than resending an identical
            # request: at temperature 0 an identical request returns an
            # identical failure.
            prompt = f"{request.prompt}\n\n{repair.render(error=last_error)}"

    raise SchemaViolationError(
        f"{request.purpose}: {request.model} failed to produce valid "
        f"{model.__name__} in {max_attempts} attempts. Last error: {last_error}"
    )
