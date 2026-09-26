"""Access control and rate limiting for a publicly reachable deployment.

Both are off by default, because the harness runs this app locally and in
tests where neither is wanted. Both are read from the environment rather than
the hydra tree: they describe the deployment, not the measured configuration,
and putting them in the config would make the config hash change when a key
rotates -- which would sever the comparability of every run recorded either
side of it.

The rate limiter is a sliding window held in process memory. That is honest
about what it is: it resets on restart and does not coordinate between
replicas, so it stops a script, not a determined adversary. A single container
behind a reverse proxy is what this repository documents, and for that it is
enough. Anything more needs shared state and belongs in front of the app.

Who "a client" is matters as much as the limit. The limiter used to key on the
first `X-Forwarded-For` entry, which the caller writes: a script that sent a
new value on every request was never limited at all, and grew the limiter's
memory by one entry per request. It now keys on the connection's peer address
unless `EVALGATE_TRUSTED_PROXY_HOPS` says how many proxies in front of the app
append to that header, and then reads the entry the outermost trusted proxy
wrote -- the only one a caller cannot forge.
"""

from __future__ import annotations

import hmac
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

API_KEY_ENV = "EVALGATE_API_KEY"
RATE_LIMIT_ENV = "EVALGATE_RATE_LIMIT_PER_MINUTE"
TRUSTED_HOPS_ENV = "EVALGATE_TRUSTED_PROXY_HOPS"
HEADER = "x-api-key"
FORWARDED_HEADER = "x-forwarded-for"
WINDOW_S = 60.0
# Idle clients are swept once the table reaches this size, and the threshold
# then doubles from what remains, so memory tracks the clients active in one
# window rather than every address ever seen.
SWEEP_AT = 1024
UNKNOWN_CLIENT = "unknown"


def configured_key() -> str | None:
    """The API key this deployment requires, if it requires one."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def key_matches(supplied: str | None, expected: str) -> bool:
    """Constant-time comparison, so a wrong key leaks no timing signal."""
    if not supplied:
        return False
    return hmac.compare_digest(supplied, expected)


def configured_rate_limit() -> int | None:
    """Requests per minute per client, if a limit is configured."""
    raw = os.environ.get(RATE_LIMIT_ENV, "").strip()
    if not raw:
        return None
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError(f"{RATE_LIMIT_ENV} must be an integer, got {raw!r}") from exc
    if limit < 1:
        raise ValueError(f"{RATE_LIMIT_ENV} must be at least 1, got {limit}")
    return limit


def configured_trusted_hops() -> int:
    """How many reverse proxies in front of the app append to X-Forwarded-For."""
    raw = os.environ.get(TRUSTED_HOPS_ENV, "").strip()
    if not raw:
        return 0
    try:
        hops = int(raw)
    except ValueError as exc:
        raise ValueError(f"{TRUSTED_HOPS_ENV} must be an integer, got {raw!r}") from exc
    if hops < 0:
        raise ValueError(f"{TRUSTED_HOPS_ENV} must not be negative, got {hops}")
    return hops


def client_identity(forwarded_for: str | None, peer: str | None, trusted_hops: int) -> str:
    """The address to rate limit: one the caller cannot choose.

    With no trusted proxy the peer address is the client. Behind `n` trusted
    proxies, each appends the address it received the request from, so the
    n-th entry from the right was written by the outermost trusted proxy and
    names the real client; anything to its left came from the caller. A header
    too short to hold that entry means the request did not come through the
    proxies at all, and the peer is used.
    """
    if trusted_hops > 0 and forwarded_for:
        entries = [entry.strip() for entry in forwarded_for.split(",") if entry.strip()]
        if len(entries) >= trusted_hops:
            return entries[-trusted_hops]
    return peer or UNKNOWN_CLIENT


@dataclass
class SlidingWindowLimiter:
    """Per-client request log over a sliding window."""

    limit: int
    window_s: float = WINDOW_S
    _hits: dict[str, deque[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)
    _sweep_at: int = SWEEP_AT

    @property
    def tracked(self) -> int:
        """How many clients the limiter currently holds state for."""
        return len(self._hits)

    def _sweep(self, cutoff: float) -> None:
        """Forget clients with nothing left in the window."""
        idle = [client for client, hits in self._hits.items() if not hits or hits[-1] <= cutoff]
        for client in idle:
            del self._hits[client]
        self._sweep_at = max(SWEEP_AT, 2 * len(self._hits))

    def allow(self, client: str, now: float | None = None) -> bool:
        """Whether this client may make a request right now."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            cutoff = moment - self.window_s
            if len(self._hits) >= self._sweep_at:
                self._sweep(cutoff)
            hits = self._hits.setdefault(client, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(moment)
            return True

    def retry_after_s(self, client: str, now: float | None = None) -> int:
        """Seconds until this client's oldest request falls out of the window."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.get(client)
            if not hits:
                return 0
            return max(1, int(self.window_s - (moment - hits[0])) + 1)
