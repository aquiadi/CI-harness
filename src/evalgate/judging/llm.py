"""The LLM judge.

Scores arrive through a forced, strict tool call validated against a pydantic
model built from the configured axes and scale. On a schema violation the model
is shown its own validation error and asked again; after the configured number
of attempts the call raises. There is no path in this class that produces a
score the model did not give.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalgate.context import format_context
from evalgate.corpus.manifest import utc_now_iso
from evalgate.evalsets.schemas import AxisScores, JudgeScore
from evalgate.judging.base import JudgeInput
from evalgate.judging.schema import build_judgment_model, split_judgment
from evalgate.models.base import ModelClient, ModelRequest
from evalgate.models.structured import build_tool, call_structured
from evalgate.prompts import Prompt

TOOL_NAME = "submit_judgment"
TOOL_DESCRIPTION = "Submit the rubric scores and their justifications for one answer."
PURPOSE = "judge"


@dataclass(slots=True)
class LLMJudge:
    """Grades answers with a model, through a validated tool call."""

    name: str
    model: str
    client: ModelClient
    prompt: Prompt
    prompts_dir: Path
    axes: list[str]
    scale_min: int
    scale_max: int
    max_tokens: int
    temperature: float
    max_attempts: int

    def score(self, item: JudgeInput) -> JudgeScore:
        """Grade one answer, or raise."""
        judgment_model = build_judgment_model(self.axes, self.scale_min, self.scale_max)
        tool = build_tool(TOOL_NAME, TOOL_DESCRIPTION, judgment_model)
        request = ModelRequest(
            model=self.model,
            prompt=self.prompt.render(
                question=item.question,
                context=format_context(item.context),
                answer=item.answer,
            ),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tool=tool,
            purpose=PURPOSE,
            prompt_hash=self.prompt.sha256,
        )
        result = call_structured(
            self.client,
            request,
            judgment_model,
            self.prompts_dir,
            max_attempts=self.max_attempts,
        )
        scores, rationales = split_judgment(result.value.model_dump(), self.axes)
        return JudgeScore(
            example_id=item.example_id,
            answer_hash=_answer_hash(item.answer),
            scores=AxisScores(**scores),
            rationales=rationales,
            judge_model=self.model,
            prompt_hash=self.prompt.sha256,
            scored_at=utc_now_iso(),
            attempts=result.attempts,
            input_tokens=result.response.input_tokens,
            output_tokens=result.response.output_tokens,
            latency_s=result.response.latency_s,
            replayed=result.response.replayed,
            variant=item.variant,
            generator=item.generator,
        )

    def fingerprint(self) -> dict[str, Any]:
        """Identity of this judge and its parameters."""
        return {
            "name": self.name,
            "model": self.model,
            "prompt_hash": self.prompt.sha256,
            "axes": list(self.axes),
            "scale": [self.scale_min, self.scale_max],
            "temperature": self.temperature,
        }


def _answer_hash(answer: str) -> str:
    from evalgate.hashing import sha256_text

    return sha256_text(answer)
