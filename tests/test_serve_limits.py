"""Access control and rate limiting, which only matter once this is public.

Every test here describes a way a deployment gets abused or taken down. The
rate limiter is a sliding window in process memory and these tests pin what that
does and does not buy -- overstating it would be worse than not having it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evalgate.serve.app import create_app
from evalgate.serve.limits import (
    API_KEY_ENV,
    HEADER,
    RATE_LIMIT_ENV,
    TRUSTED_HOPS_ENV,
    SlidingWindowLimiter,
    client_identity,
    configured_key,
    configured_rate_limit,
    configured_trusted_hops,
    key_matches,
)

QUESTION = {"question": "When is the quarterly CBAM report due?"}


def app_overrides(tmp_path: Path) -> list[str]:
    """The fixture corpus and a dependency-free embedder, as test_serve uses."""
    root = Path(__file__).resolve().parent.parent
    return [
        "+experiment=baseline",
        "embedder=hashed",
        "corpus.name=fixture",
        f"corpus.local_dir={root / 'tests' / 'fixtures' / 'corpus'}",
        "chunker.target_tokens=96",
        f"paths.index_dir={tmp_path / 'index'}",
    ]


def test_no_key_configured_means_no_authentication() -> None:
    """Local runs and the test suite must not need a key."""
    assert configured_key() is None


def test_a_blank_key_is_not_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty env var must not silently enable auth that rejects everyone."""
    monkeypatch.setenv(API_KEY_ENV, "   ")
    assert configured_key() is None


def test_key_comparison_rejects_the_obvious_wrong_answers() -> None:
    assert key_matches("secret", "secret")
    assert not key_matches("secre", "secret")
    assert not key_matches("", "secret")
    assert not key_matches(None, "secret")


def test_a_bad_rate_limit_is_refused_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Better to fail to boot than to serve with the limiter silently off."""
    monkeypatch.setenv(RATE_LIMIT_ENV, "not-a-number")
    with pytest.raises(ValueError, match="must be an integer"):
        configured_rate_limit()
    monkeypatch.setenv(RATE_LIMIT_ENV, "0")
    with pytest.raises(ValueError, match="at least 1"):
        configured_rate_limit()


def test_the_limiter_allows_up_to_the_limit_then_stops() -> None:
    limiter = SlidingWindowLimiter(limit=3, window_s=60.0)
    assert [limiter.allow("a", now=100.0) for _ in range(4)] == [True, True, True, False]


def test_clients_are_counted_separately() -> None:
    """One noisy caller must not lock everyone else out."""
    limiter = SlidingWindowLimiter(limit=1, window_s=60.0)
    assert limiter.allow("a", now=100.0)
    assert not limiter.allow("a", now=100.0)
    assert limiter.allow("b", now=100.0)


def test_the_window_rolls_forward() -> None:
    limiter = SlidingWindowLimiter(limit=1, window_s=60.0)
    assert limiter.allow("a", now=100.0)
    assert not limiter.allow("a", now=159.0)
    assert limiter.allow("a", now=161.0)


def test_retry_after_is_never_zero_while_blocked() -> None:
    """A retry-after of 0 invites an immediate retry that fails again."""
    limiter = SlidingWindowLimiter(limit=1, window_s=60.0)
    limiter.allow("a", now=100.0)
    assert limiter.retry_after_s("a", now=100.0) >= 1


def test_query_requires_the_key_when_one_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "s3cret")
    with TestClient(create_app(app_overrides(tmp_path))) as client:
        assert client.post("/query", json=QUESTION).status_code == 401
        assert client.post("/query", json=QUESTION, headers={HEADER: "wrong"}).status_code == 401


def test_health_stays_open_so_probes_and_balancers_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A readiness probe has no API key; gating it takes the service down."""
    monkeypatch.setenv(API_KEY_ENV, "s3cret")
    monkeypatch.setenv(RATE_LIMIT_ENV, "1")
    with TestClient(create_app(app_overrides(tmp_path))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200


def test_the_rate_limit_returns_429_with_a_retry_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(RATE_LIMIT_ENV, "1")
    with TestClient(create_app(app_overrides(tmp_path))) as client:
        assert client.post("/query", json=QUESTION).status_code == 200
        blocked = client.post("/query", json=QUESTION)
        assert blocked.status_code == 429
        assert int(blocked.headers["retry-after"]) >= 1


def test_the_ui_is_served_and_names_no_secret(tmp_path: Path) -> None:
    with TestClient(create_app(app_overrides(tmp_path))) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "evalgate" in page.text
        assert client.get("/static/app.js").status_code == 200


def test_a_forged_forwarded_for_does_not_escape_the_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A new X-Forwarded-For per request used to be a new client per request."""
    monkeypatch.setenv(RATE_LIMIT_ENV, "1")
    with TestClient(create_app(app_overrides(tmp_path))) as client:
        first = client.post("/query", json=QUESTION, headers={"x-forwarded-for": "1.1.1.1"})
        forged = client.post("/query", json=QUESTION, headers={"x-forwarded-for": "2.2.2.2"})
        assert first.status_code == 200
        assert forged.status_code == 429


def test_without_trusted_proxies_the_peer_is_the_client() -> None:
    assert client_identity("6.6.6.6", "10.0.0.5", trusted_hops=0) == "10.0.0.5"


def test_behind_a_trusted_proxy_only_the_entry_it_wrote_counts() -> None:
    """Everything left of the proxy's own entry came from the caller."""
    assert client_identity("6.6.6.6, 203.0.113.9", "10.0.0.1", trusted_hops=1) == "203.0.113.9"
    assert (
        client_identity("6.6.6.6, 203.0.113.9, 10.0.0.2", "10.0.0.1", trusted_hops=2)
        == "203.0.113.9"
    )


def test_a_header_too_short_for_the_proxies_falls_back_to_the_peer() -> None:
    assert client_identity("203.0.113.9", "10.0.0.1", trusted_hops=2) == "10.0.0.1"
    assert client_identity(None, None, trusted_hops=1) == "unknown"


def test_a_bad_proxy_hop_count_is_refused_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRUSTED_HOPS_ENV, "two")
    with pytest.raises(ValueError, match="must be an integer"):
        configured_trusted_hops()
    monkeypatch.setenv(TRUSTED_HOPS_ENV, "-1")
    with pytest.raises(ValueError, match="must not be negative"):
        configured_trusted_hops()


def test_idle_clients_are_forgotten_so_memory_stays_bounded() -> None:
    """One entry per distinct address forever is a memory leak with extra steps."""
    limiter = SlidingWindowLimiter(limit=5, window_s=60.0)
    # One new address a second for 5000 seconds: about sixty are active at any
    # moment, and the table must track roughly those, not all 5000.
    for index in range(5000):
        limiter.allow(f"client-{index}", now=float(index))
    assert limiter.tracked <= 1024 + 60
