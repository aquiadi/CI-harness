"""Access control and rate limiting for a publicly reachable deployment.

Both are off by default, because the harness runs this app locally and in
tests where neither is wanted. Both are read from the environment rather than
the hydra tree: they describe the deployment, not the measured configuration,
and putting them in the config would make the config hash change when a key
rotates -- which would sever the comparability of every run recorded either
side of it.

The rate limiter is a fixed-window counter held in process memory. That is
honest about what it is: it resets on restart and does not coordinate between
replicas, so it stops a script, not a determined adversary. A single container
behind a reverse proxy is what this repository documents, and for that it is
enough. Anything more needs shared state and belongs in front of the app.
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
HEADER = "x-api-key"
WINDOW_S = 60.0


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


@dataclass
class FixedWindowLimiter:
    """Per-client request counter over a rolling window."""

    limit: int
    window_s: float = WINDOW_S
    _hits: dict[str, deque[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def allow(self, client: str, now: float | None = None) -> bool:
        """Whether this client may make a request right now."""
        moment = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            cutoff = moment - self.window_s
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
