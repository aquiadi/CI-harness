"""The FastAPI layer.

Deliberately thin. It builds the same stack the harness measures, through the
same factory functions, and adds nothing to the answer path. Anything it added
would be untested by the evaluation: a serving layer that reranks, caches or
rewrites queries on its own is a different system from the one the Pareto table
describes, and the table would quietly stop being true.

Configuration comes from the same hydra tree. `EVALGATE_OVERRIDES` passes
overrides in, space separated, so a container can be pointed at a different
corpus or generator without a rebuild:

    EVALGATE_OVERRIDES="+experiment=live" uvicorn evalgate.serve.app:app
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from omegaconf import DictConfig

from evalgate.citations import audit_citations
from evalgate.config import (
    GeneratorConfig,
    PricingConfig,
    config_hash,
    load_config,
    resolve_path,
    typed_node,
)
from evalgate.context import context_chunk_ids
from evalgate.errors import EvalgateError
from evalgate.generation.base import GenerationInput, Generator
from evalgate.generation.factory import build_generator
from evalgate.models.client import build_model_client
from evalgate.pipeline import RetrievalStack, build_stack
from evalgate.prompts import load_prompt
from evalgate.serve.schemas import (
    Citation,
    Cost,
    Health,
    Latency,
    Provenance,
    QueryRequest,
    QueryResponse,
    RetrievalTrace,
    RetrievedChunk,
)

OVERRIDES_ENV = "EVALGATE_OVERRIDES"
USD_PER_MTOK = 1_000_000.0
PROVIDER_EXTRACTIVE = "extractive"
TITLE = "evalgate"
DESCRIPTION = "Retrieval-augmented answers over CBAM regulatory documents, with their trace."


@dataclass(slots=True)
class Service:
    """The built stack, held for the process lifetime."""

    cfg: DictConfig
    stack: RetrievalStack
    generator: Generator
    generator_cfg: GeneratorConfig
    pricing: PricingConfig
    config_hash: str
    generator_prompt_hash: str


def overrides_from_env() -> list[str]:
    """Hydra overrides passed through the environment."""
    return os.environ.get(OVERRIDES_ENV, "").split()


def build_service(overrides: list[str] | None = None) -> Service:
    """Compose the config and build the stack once."""
    cfg = load_config(overrides=overrides if overrides is not None else overrides_from_env())
    generator_cfg = typed_node(cfg, "generator", GeneratorConfig)
    client = None if generator_cfg.provider == PROVIDER_EXTRACTIVE else build_model_client(cfg)
    stack = build_stack(cfg)
    prompt = load_prompt(resolve_path(cfg, "paths.prompts_dir"), generator_cfg.prompt)
    return Service(
        cfg=cfg,
        stack=stack,
        generator=build_generator(cfg, stack.tokenizer, client),
        generator_cfg=generator_cfg,
        pricing=typed_node(cfg, "pricing", PricingConfig),
        config_hash=config_hash(cfg),
        generator_prompt_hash=prompt.sha256,
    )


def answer_question(service: Service, question: str, k: int | None) -> QueryResponse:
    """Retrieve, generate, and report what it took."""
    started = time.perf_counter()
    results = service.stack.retriever.retrieve(question, k=k)
    retrieval_latency = time.perf_counter() - started

    generated = service.generator.generate(
        GenerationInput(example_id="serve", question=question, context=results)
    )
    audit = audit_citations(generated.answer, context_chunk_ids(results))
    answer_tokens = generated.output_tokens or service.stack.tokenizer.count(generated.answer)

    return QueryResponse(
        answer=generated.answer,
        citations=[
            Citation(chunk_id=chunk_id, valid=chunk_id in set(audit.valid))
            for chunk_id in audit.cited
        ],
        retrieval=RetrievalTrace(
            retriever=str(service.cfg.retriever.name),
            k=len(results),
            index_hash=service.stack.index.meta.index_hash,
            chunks=[
                RetrievedChunk(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    doc_title=result.doc_title,
                    section=result.section,
                    rank=result.rank,
                    score=result.score,
                )
                for result in results
            ],
            latency_s=retrieval_latency,
        ),
        cost=Cost(
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            context_tokens=generated.context_tokens,
            usd=generated.cost_usd(
                service.generator_cfg.input_usd_per_mtok,
                service.generator_cfg.output_usd_per_mtok,
            ),
            projected_usd=(
                generated.context_tokens * service.pricing.input_usd_per_mtok
                + answer_tokens * service.pricing.output_usd_per_mtok
            )
            / USD_PER_MTOK,
        ),
        latency=Latency(
            retrieval_s=retrieval_latency,
            generation_s=generated.latency_s,
            total_s=retrieval_latency + generated.latency_s,
        ),
        provenance=Provenance(
            corpus=service.stack.corpus.name,
            corpus_hash=service.stack.corpus.corpus_hash,
            index_hash=service.stack.index.meta.index_hash,
            generator_model=generated.model,
            generator_prompt_hash=service.generator_prompt_hash,
            config_hash=service.config_hash,
            replayed=generated.replayed,
        ),
    )


def create_app(overrides: list[str] | None = None) -> FastAPI:
    """Build the application, with the stack constructed at startup."""
    state: dict[str, Service] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Building the index at startup rather than per request: the first
        # request should not pay for it, and a container that cannot build it
        # should fail to start rather than fail on traffic.
        state["service"] = build_service(overrides)
        yield
        state.clear()

    app = FastAPI(title=TITLE, description=DESCRIPTION, lifespan=lifespan)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        """Readiness, and what this process is serving."""
        service = state["service"]
        return Health(
            status="ok",
            corpus=service.stack.corpus.name,
            corpus_hash=service.stack.corpus.corpus_hash,
            index_hash=service.stack.index.meta.index_hash,
            chunks=service.stack.index.size,
            retriever=str(service.cfg.retriever.name),
            generator_model=service.generator.model,
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        """Answer one question, with its retrieval trace and cost."""
        service = state["service"]
        try:
            return answer_question(service, request.question, request.k)
        except EvalgateError as exc:
            # Errors this codebase raises on purpose are the caller's problem to
            # understand: a missing cassette, an unavailable backend. Anything
            # else is a bug and gets the default 500 with no detail.
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def _app() -> Any:
    return create_app()


app = _app()
