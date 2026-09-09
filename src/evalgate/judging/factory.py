"""Selecting a judge from config."""

from __future__ import annotations

from typing import cast

from omegaconf import DictConfig

from evalgate.config import JudgeConfig, resolve_path, typed_node
from evalgate.judging.base import Judge
from evalgate.judging.heuristic import HeuristicJudge
from evalgate.judging.llm import LLMJudge
from evalgate.models.base import ModelClient
from evalgate.prompts import load_prompt

PROVIDER_HEURISTIC = "heuristic"
# "api" means "an LLM reached through the model client". Which vendor actually
# answers is api.provider, because the judge does not care -- it takes a
# ModelClient. "anthropic" is the original spelling of this value and is kept
# as an alias so existing configs and run records keep working.
MODEL_CLIENT_PROVIDERS = frozenset({"api", "anthropic"})


def build_judge(cfg: DictConfig, client: ModelClient | None = None) -> Judge:
    """Instantiate the configured judge.

    The heuristic baseline needs no client, which is what lets the calibration
    report run with no credentials and no cassettes.
    """
    settings = typed_node(cfg, "judge", JudgeConfig)
    if settings.provider == PROVIDER_HEURISTIC:
        return cast(
            Judge,
            HeuristicJudge(
                name=PROVIDER_HEURISTIC,
                model=settings.model,
                axes=list(settings.axes),
                scale_min=settings.scale_min,
                scale_max=settings.scale_max,
            ),
        )
    if settings.provider not in MODEL_CLIENT_PROVIDERS:
        raise ValueError(
            f"unknown judge provider {settings.provider!r}; expected "
            f"{PROVIDER_HEURISTIC!r} or one of {sorted(MODEL_CLIENT_PROVIDERS)}"
        )
    if client is None:
        raise ValueError("an LLM judge needs a model client")

    prompts_dir = resolve_path(cfg, "paths.prompts_dir")
    return cast(
        Judge,
        LLMJudge(
            name=settings.provider,
            model=settings.model,
            client=client,
            prompt=load_prompt(prompts_dir, settings.prompt),
            prompts_dir=prompts_dir,
            axes=list(settings.axes),
            scale_min=settings.scale_min,
            scale_max=settings.scale_max,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            reasoning_effort=settings.reasoning_effort,
            max_attempts=settings.max_attempts,
        ),
    )
