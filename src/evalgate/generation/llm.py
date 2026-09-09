"""The API generator: the system as it is meant to run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evalgate.context import format_context
from evalgate.generation.base import Generated, GenerationInput
from evalgate.ingest.tokens import TokenEstimator
from evalgate.models.base import ModelClient, ModelRequest
from evalgate.prompts import Prompt

PURPOSE = "generation"


@dataclass(slots=True)
class LLMGenerator:
    """Answers with a model, over the retrieved context and nothing else."""

    name: str
    model: str
    client: ModelClient
    prompt: Prompt
    tokenizer: TokenEstimator
    max_tokens: int
    temperature: float
    reasoning_effort: str | None = None

    def generate(self, item: GenerationInput) -> Generated:
        """Answer one question."""
        context = format_context(item.context)
        response = self.client.complete(
            ModelRequest(
                model=self.model,
                prompt=self.prompt.render(context=context, question=item.question),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                purpose=PURPOSE,
                prompt_hash=self.prompt.sha256,
            )
        )
        return Generated(
            example_id=item.example_id,
            answer=response.text,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_s=response.latency_s,
            context_tokens=self.tokenizer.count(context),
            replayed=response.replayed,
        )

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this generator and its parameters."""
        return {
            "name": self.name,
            "model": self.model,
            "prompt_hash": self.prompt.sha256,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
