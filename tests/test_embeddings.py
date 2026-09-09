from __future__ import annotations

import numpy as np
import pytest
from omegaconf import DictConfig

from evalgate.config import load_config
from evalgate.embeddings.base import Embedder, l2_normalise
from evalgate.embeddings.local import BackendUnavailableError, SentenceTransformerEmbedder
from evalgate.pipeline import build_embedder

TEXTS = ["quarterly reporting obligation", "embedded emissions of imported cement"]


@pytest.fixture
def hashed() -> Embedder:
    cfg: DictConfig = load_config(overrides=["embedder=hashed"])
    return build_embedder(cfg)


def test_document_matrix_has_the_declared_shape(hashed: Embedder) -> None:
    assert hashed.embed_documents(TEXTS).shape == (len(TEXTS), hashed.dim)


def test_query_vector_has_the_declared_shape(hashed: Embedder) -> None:
    assert hashed.embed_query(TEXTS[0]).shape == (hashed.dim,)


def test_empty_input_returns_an_empty_matrix(hashed: Embedder) -> None:
    assert hashed.embed_documents([]).shape == (0, hashed.dim)


def test_vectors_are_unit_length(hashed: Embedder) -> None:
    norms = np.linalg.norm(hashed.embed_documents(TEXTS), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_embedding_is_deterministic_across_instances() -> None:
    cfg: DictConfig = load_config(overrides=["embedder=hashed"])
    first = build_embedder(cfg).embed_documents(TEXTS)
    second = build_embedder(cfg).embed_documents(TEXTS)
    assert np.array_equal(first, second)


def test_identical_text_embeds_identically(hashed: Embedder) -> None:
    matrix = hashed.embed_documents(["same text", "same text"])
    assert np.array_equal(matrix[0], matrix[1])


def test_related_text_scores_above_unrelated(hashed: Embedder) -> None:
    query = hashed.embed_query("quarterly reporting obligation")
    matrix = hashed.embed_documents(
        ["the quarterly reporting obligation applies", "unrelated culinary instructions"]
    )
    assert float(matrix[0] @ query) > float(matrix[1] @ query)


def test_seed_changes_the_projection() -> None:
    a = build_embedder(load_config(overrides=["embedder=hashed"]))
    b = build_embedder(load_config(overrides=["embedder=hashed", "embedder.seed=7"]))
    assert not np.array_equal(a.embed_query(TEXTS[0]), b.embed_query(TEXTS[0]))
    assert a.fingerprint() != b.fingerprint()


def test_l2_normalise_leaves_zero_rows_alone() -> None:
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[1, 0] = 3.0
    normalised = l2_normalise(matrix)
    assert np.array_equal(normalised[0], np.zeros(3, dtype=np.float32))
    assert np.isclose(float(np.linalg.norm(normalised[1])), 1.0)


def test_local_backend_reports_a_useful_error_without_the_extra() -> None:
    """The default embedder needs `make ml`; the failure must say so."""
    embedder = SentenceTransformerEmbedder(
        name="local",
        model="BAAI/bge-small-en-v1.5",
        dim=384,
        batch_size=8,
        normalize=True,
        query_prefix="",
        document_prefix="",
        device="cpu",
    )
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(BackendUnavailableError, match="make ml"):
            embedder.embed_query("x")
