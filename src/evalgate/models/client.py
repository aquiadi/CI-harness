"""Composing the model client from config.

Three modes, one interface:

    replay   cassettes only; an unmatched call is an error. PR CI runs here.
    record   live calls, written to cassettes as they happen.
    live     live calls, nothing written. The nightly workflow runs here.

The sqlite cache sits inside the recorder so that re-running a record session
does not re-pay for calls already made, while still writing every cassette.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from evalgate.config import ApiConfig, resolve_path, typed_node
from evalgate.models.base import ModelClient, ModelRequest, ModelResponse
from evalgate.models.cache import ResponseCache
from evalgate.models.cassette import CassetteMissError, CassetteStore

MODE_REPLAY = "replay"
MODE_RECORD = "record"
MODE_LIVE = "live"
MODES = (MODE_REPLAY, MODE_RECORD, MODE_LIVE)


@dataclass(slots=True)
class ReplayClient:
    """Serves recorded responses and refuses to reach the network."""

    store: CassetteStore
    name: str = "replay"

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the recorded response, or fail with how to record it."""
        recorded = self.store.load(request)
        if recorded is None:
            raise CassetteMissError(
                f"no cassette for {request.purpose} call to {request.model} "
                f"(key {request.cache_key()[:16]}) in {self.store.path}. "
                "Re-record with `make eval ARGS=api=record` and commit the result; "
                "a miss usually means a prompt, model or config change."
            )
        return recorded


@dataclass(slots=True)
class CachingClient:
    """Serves from the local sqlite cache before calling the inner client."""

    inner: ModelClient
    cache: ResponseCache
    name: str = "cached"

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a cached response, else call through and store the result."""
        hit = self.cache.get(request)
        if hit is not None:
            return hit
        response = self.inner.complete(request)
        self.cache.put(request, response)
        return response


@dataclass(slots=True)
class RecordingClient:
    """Writes a cassette for every call that passes through."""

    inner: ModelClient
    store: CassetteStore
    name: str = "recording"

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Call through, then persist the exchange."""
        response = self.inner.complete(request)
        self.store.save(request, response)
        return response


def build_model_client(cfg: DictConfig, cassette_dir: Path | None = None) -> ModelClient:
    """Wire the client stack for the configured API mode."""
    api = typed_node(cfg, "api", ApiConfig)
    if api.mode not in MODES:
        raise ValueError(f"api.mode must be one of {MODES}, got {api.mode!r}")

    store = CassetteStore(cassette_dir or resolve_path(cfg, "paths.cassette_dir"))
    if api.mode == MODE_REPLAY:
        return ReplayClient(store=store)

    from evalgate.models.anthropic_client import AnthropicClient

    live: ModelClient = AnthropicClient(
        timeout_s=api.timeout_s,
        max_retries=api.max_retries,
        backoff_initial_s=api.backoff_initial_s,
        backoff_max_s=api.backoff_max_s,
    )
    if api.cache_enabled:
        live = CachingClient(inner=live, cache=ResponseCache(resolve_path(cfg, "api.cache_path")))
    if api.mode == MODE_RECORD:
        return RecordingClient(inner=live, store=store)
    return live
