from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evalgate.serve.app import create_app, overrides_from_env


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    root = Path(__file__).resolve().parent.parent
    overrides = [
        "+experiment=baseline",
        "corpus.name=fixture",
        f"corpus.local_dir={root / 'tests' / 'fixtures' / 'corpus'}",
        "chunker.target_tokens=96",
        f"paths.index_dir={tmp_path_factory.mktemp('serve-index')}",
    ]
    with TestClient(create_app(overrides)) as running:
        yield running


def test_health_reports_what_is_being_served(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["corpus"] == "fixture"
    assert payload["chunks"] > 0
    assert payload["index_hash"]


def test_query_returns_an_answer_with_a_retrieval_trace(client: TestClient) -> None:
    response = client.post("/query", json={"question": "When is the quarterly report due?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["retrieval"]["chunks"]
    assert payload["retrieval"]["k"] == len(payload["retrieval"]["chunks"])
    assert payload["retrieval"]["latency_s"] >= 0


def test_retrieval_trace_is_ordered_by_rank(client: TestClient) -> None:
    chunks = client.post("/query", json={"question": "quarterly report"}).json()["retrieval"][
        "chunks"
    ]
    assert [chunk["rank"] for chunk in chunks] == list(range(len(chunks)))


def test_citations_are_reported_with_their_validity(client: TestClient) -> None:
    payload = client.post("/query", json={"question": "When is the quarterly report due?"}).json()
    assert payload["citations"]
    assert all(citation["valid"] for citation in payload["citations"])
    available = {chunk["chunk_id"] for chunk in payload["retrieval"]["chunks"]}
    assert {citation["chunk_id"] for citation in payload["citations"]} <= available


def test_cost_and_latency_are_reported(client: TestClient) -> None:
    payload = client.post("/query", json={"question": "quarterly report"}).json()
    assert payload["cost"]["context_tokens"] > 0
    assert payload["cost"]["usd"] == 0.0  # the extractive generator makes no API call
    assert payload["cost"]["projected_usd"] > 0.0
    assert payload["latency"]["total_s"] == pytest.approx(
        payload["latency"]["retrieval_s"] + payload["latency"]["generation_s"]
    )


def test_provenance_identifies_what_answered(client: TestClient) -> None:
    provenance = client.post("/query", json={"question": "quarterly report"}).json()["provenance"]
    assert provenance["corpus"] == "fixture"
    assert len(provenance["corpus_hash"]) == 64
    assert len(provenance["generator_prompt_hash"]) == 64
    assert provenance["config_hash"]


def test_k_can_be_overridden_per_request(client: TestClient) -> None:
    payload = client.post("/query", json={"question": "quarterly report", "k": 2}).json()
    assert len(payload["retrieval"]["chunks"]) == 2


def test_an_empty_question_is_rejected(client: TestClient) -> None:
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_unknown_fields_are_rejected(client: TestClient) -> None:
    response = client.post("/query", json={"question": "q", "temperature": 0.9})
    assert response.status_code == 422


def test_an_out_of_range_k_is_rejected(client: TestClient) -> None:
    assert client.post("/query", json={"question": "q", "k": 0}).status_code == 422


def test_overrides_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALGATE_OVERRIDES", "+experiment=live retriever.k=3")
    assert overrides_from_env() == ["+experiment=live", "retriever.k=3"]


def test_no_overrides_is_an_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALGATE_OVERRIDES", raising=False)
    assert overrides_from_env() == []
