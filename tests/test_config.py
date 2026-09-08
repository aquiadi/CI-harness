from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from evalgate.config import (
    CorpusConfig,
    GateConfig,
    JudgeConfig,
    config_hash,
    fingerprint,
    load_config,
    resolve_path,
    typed_node,
)


def test_default_config_composes(cfg: DictConfig) -> None:
    assert cfg.seed == 0
    assert cfg.generator.model
    assert cfg.judge.model
    assert cfg.retriever.k > 0


def test_all_paths_resolve_absolute(cfg: DictConfig, repo_root: Path) -> None:
    for key in cfg.paths:
        path = resolve_path(cfg, f"paths.{key}")
        assert path.is_absolute()
        assert path.is_relative_to(repo_root)


def test_typed_node_returns_dataclass(cfg: DictConfig) -> None:
    corpus = typed_node(cfg, "corpus", CorpusConfig)
    assert corpus.name == "cbam"
    assert corpus.sources
    assert all(source.url.startswith("https://") for source in corpus.sources)


def test_typed_node_rejects_wrong_type(cfg: DictConfig) -> None:
    with pytest.raises(TypeError):
        typed_node(cfg, "corpus", JudgeConfig)


def test_config_hash_is_deterministic() -> None:
    assert config_hash(load_config()) == config_hash(load_config())


def test_config_hash_changes_with_retriever(cfg: DictConfig) -> None:
    other = load_config(overrides=["retriever=bm25"])
    assert config_hash(other) != config_hash(cfg)


def test_config_hash_changes_with_k(cfg: DictConfig) -> None:
    other = load_config(overrides=["retriever.k=3"])
    assert config_hash(other) != config_hash(cfg)


def test_config_hash_ignores_api_mode(cfg: DictConfig) -> None:
    """Replaying a call must not change the identity of the measurement."""
    other = load_config(overrides=["api=live"])
    assert config_hash(other) == config_hash(cfg)


def test_fingerprint_excludes_paths_and_gate(cfg: DictConfig) -> None:
    keys = set(fingerprint(cfg))
    assert "paths" not in keys
    assert "gate" not in keys
    assert "api" not in keys
    assert {"chunker", "retriever", "embedder", "judge", "generator"} <= keys


def test_fingerprint_keeps_corpus_identity_not_fetch_mechanics(cfg: DictConfig) -> None:
    corpus = fingerprint(cfg)["corpus"]
    assert set(corpus) == {"name", "sources"}


def test_judge_is_swappable_to_a_different_model_than_the_generator() -> None:
    cfg = load_config(overrides=["judge=opus"])
    judge = typed_node(cfg, "judge", JudgeConfig)
    assert judge.model != cfg.generator.model
    assert judge.axes == ["groundedness", "relevance", "citation_correctness"]


def test_gate_composite_weights_sum_to_one(cfg: DictConfig) -> None:
    gate = typed_node(cfg, "gate", GateConfig)
    assert sum(gate.composite_weights.values()) == pytest.approx(1.0)


def test_every_chunker_and_retriever_config_composes() -> None:
    for chunker in ("fixed_token", "recursive_structural", "section_aware"):
        assert load_config(overrides=[f"chunker={chunker}"]).chunker.name == chunker
    for retriever in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        assert load_config(overrides=[f"retriever={retriever}"]).retriever.name == retriever
    for embedder in ("local", "api", "hashed"):
        assert load_config(overrides=[f"embedder={embedder}"]).embedder.name == embedder


def test_pluggable_components_declare_a_target(cfg: DictConfig) -> None:
    for group in ("chunker", "embedder", "retriever"):
        node = OmegaConf.select(cfg, group)
        assert "_target_" in node, group


def test_unknown_key_override_is_rejected() -> None:
    with pytest.raises(Exception, match=r"not found|Could not override"):
        load_config(overrides=["retriever.nonexistent_key=1"])
