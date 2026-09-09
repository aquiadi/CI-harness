"""Selecting a generator from config."""

from __future__ import annotations

from typing import cast

from omegaconf import DictConfig

from evalgate.config import GeneratorConfig, resolve_path, typed_node
from evalgate.generation.base import Generator
from evalgate.generation.extractive import ExtractiveGenerator
from evalgate.generation.llm import LLMGenerator
from evalgate.ingest.tokens import TokenEstimator
from evalgate.models.base import ModelClient
from evalgate.prompts import load_prompt

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_EXTRACTIVE = "extractive"


def build_generator(
    cfg: DictConfig, tokenizer: TokenEstimator, client: ModelClient | None = None
) -> Generator:
    """Instantiate the configured generator."""
    settings = typed_node(cfg, "generator", GeneratorConfig)
    if settings.provider == PROVIDER_EXTRACTIVE:
        return cast(
            Generator,
            ExtractiveGenerator(
                name=PROVIDER_EXTRACTIVE,
                model=settings.model,
                tokenizer=tokenizer,
                max_sentences=settings.max_sentences,
                max_chunks=settings.max_chunks,
                min_overlap=settings.min_overlap,
            ),
        )
    if settings.provider != PROVIDER_ANTHROPIC:
        raise ValueError(f"unknown generator provider {settings.provider!r}")
    if client is None:
        raise ValueError("an LLM generator needs a model client")
    return cast(
        Generator,
        LLMGenerator(
            name=settings.provider,
            model=settings.model,
            client=client,
            prompt=load_prompt(resolve_path(cfg, "paths.prompts_dir"), settings.prompt),
            tokenizer=tokenizer,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        ),
    )
