from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from omegaconf import DictConfig

from evalgate.config import load_config
from evalgate.pipeline import build_embedder, build_retriever, build_stack
from evalgate.retrieval.base import Retriever
from evalgate.retrieval.index import ChunkIndex

RETRIEVERS = ["dense", "bm25", "hybrid"]
QUERIES = [
    "When must the quarterly report be submitted?",
    "authorised declarant application decision period",
    "How are embedded emissions calculated?",
    "electricity megawatt hours",
    "zzzz nonexistent vocabulary qqqq",
]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[DictConfig, ChunkIndex]:
    """One index shared by every retriever test: building it is the slow part."""
    root = Path(__file__).resolve().parent.parent
    cfg = load_config(
        overrides=[
            "corpus.name=fixture",
            f"corpus.local_dir={root / 'tests' / 'fixtures' / 'corpus'}",
            "embedder=hashed",
            "chunker.target_tokens=96",
            f"paths.index_dir={tmp_path_factory.mktemp('index')}",
        ]
    )
    return cfg, build_stack(cfg).index


def retriever_for(name: str, built: tuple[DictConfig, ChunkIndex], k: int = 5) -> Retriever:
    cfg, index = built
    overrides = dict(cfg)
    variant = load_config(
        overrides=[
            f"retriever={name}",
            f"retriever.k={k}",
            "embedder=hashed",
            f"corpus.local_dir={overrides['corpus']['local_dir']}",
        ]
    )
    return build_retriever(variant, index, build_embedder(variant))


@pytest.mark.parametrize("name", RETRIEVERS)
@pytest.mark.parametrize("query", QUERIES)
def test_returns_exactly_k_results(
    name: str, query: str, built: tuple[DictConfig, ChunkIndex]
) -> None:
    _, index = built
    results = retriever_for(name, built, k=5).retrieve(query)
    assert len(results) == min(5, index.size)


@pytest.mark.parametrize("name", RETRIEVERS)
@pytest.mark.parametrize("query", QUERIES)
def test_scores_are_monotonically_non_increasing(
    name: str, query: str, built: tuple[DictConfig, ChunkIndex]
) -> None:
    scores = [result.score for result in retriever_for(name, built).retrieve(query)]
    assert all(earlier >= later for earlier, later in pairwise(scores))


@pytest.mark.parametrize("name", RETRIEVERS)
@pytest.mark.parametrize("query", QUERIES)
def test_ordering_is_stable_under_repeated_calls(
    name: str, query: str, built: tuple[DictConfig, ChunkIndex]
) -> None:
    retriever = retriever_for(name, built)
    first = retriever.retrieve(query)
    for _ in range(3):
        assert retriever.retrieve(query) == first


@pytest.mark.parametrize("name", RETRIEVERS)
def test_two_instances_agree(built: tuple[DictConfig, ChunkIndex], name: str) -> None:
    """A fresh retriever over the same index must not reorder anything."""
    query = QUERIES[0]
    assert retriever_for(name, built).retrieve(query) == retriever_for(name, built).retrieve(query)


@pytest.mark.parametrize("name", RETRIEVERS)
@pytest.mark.parametrize("query", QUERIES)
def test_results_are_distinct_and_indexed(
    name: str, query: str, built: tuple[DictConfig, ChunkIndex]
) -> None:
    _, index = built
    results = retriever_for(name, built).retrieve(query)
    ids = [result.chunk_id for result in results]
    assert len(set(ids)) == len(ids)
    assert all(chunk_id in index.chunks for chunk_id in ids)
    assert [result.rank for result in results] == list(range(len(results)))


@pytest.mark.parametrize("name", RETRIEVERS)
def test_k_argument_overrides_configured_k(built: tuple[DictConfig, ChunkIndex], name: str) -> None:
    retriever = retriever_for(name, built, k=5)
    assert len(retriever.retrieve(QUERIES[0], k=2)) == 2


@pytest.mark.parametrize("name", RETRIEVERS)
def test_k_above_index_size_is_clamped(built: tuple[DictConfig, ChunkIndex], name: str) -> None:
    _, index = built
    assert len(retriever_for(name, built).retrieve(QUERIES[0], k=index.size + 50)) == index.size


@pytest.mark.parametrize("name", RETRIEVERS)
@settings(
    max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(query=st.text(alphabet=st.characters(codec="ascii"), min_size=1, max_size=120))
def test_invariants_hold_for_arbitrary_queries(
    name: str, query: str, built: tuple[DictConfig, ChunkIndex]
) -> None:
    _, index = built
    results = retriever_for(name, built, k=4).retrieve(query)
    assert len(results) == min(4, index.size)
    scores = [result.score for result in results]
    assert all(a >= b for a, b in pairwise(scores))
    assert len({result.chunk_id for result in results}) == len(results)


@pytest.mark.parametrize("name", RETRIEVERS)
@pytest.mark.parametrize("query", [":", "", "   ", "!!!"])
def test_signal_free_queries_still_return_k_deterministically(
    name: str, query: str, built: tuple[DictConfig, ChunkIndex]
) -> None:
    """A query with no indexable content still returns k, uniformly.

    Found by a hypothesis property test: LanceDB dropped the undefined cosine
    distances and returned nothing while BM25 returned k zero-scored rows.
    """
    _, index = built
    results = retriever_for(name, built, k=4).retrieve(query)
    assert len(results) == min(4, index.size)
    assert results == retriever_for(name, built, k=4).retrieve(query)


def test_bm25_finds_an_exact_rare_term(built: tuple[DictConfig, ChunkIndex]) -> None:
    results = retriever_for("bm25", built).retrieve("megawatt hours")
    assert any("megawatt" in result.text.lower() for result in results)


def test_hybrid_keeps_what_both_arms_agree_on(built: tuple[DictConfig, ChunkIndex]) -> None:
    query = "quarterly report submission deadline"
    dense = {result.chunk_id for result in retriever_for("dense", built).retrieve(query)}
    sparse = {result.chunk_id for result in retriever_for("bm25", built).retrieve(query)}
    hybrid = {result.chunk_id for result in retriever_for("hybrid", built).retrieve(query)}
    assert dense & sparse <= hybrid


def test_fingerprints_distinguish_retrievers(built: tuple[DictConfig, ChunkIndex]) -> None:
    prints = [retriever_for(name, built).fingerprint() for name in RETRIEVERS]
    assert len({str(sorted(p.items())) for p in prints}) == len(prints)
