from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from evalgate.config import load_config
from evalgate.pipeline import build_stack
from evalgate.retrieval.index import IndexBuildError, build_index, load_index

SENTINEL = "sentinel.txt"


def config_for(corpus_dir: Path, index_dir: Path, *overrides: str) -> DictConfig:
    return load_config(
        overrides=[
            "corpus.name=fixture",
            f"corpus.local_dir={corpus_dir}",
            "embedder=hashed",
            "chunker.target_tokens=96",
            f"paths.index_dir={index_dir}",
            *overrides,
        ]
    )


def test_rebuilding_an_unchanged_corpus_is_a_no_op(
    fixture_corpus_dir: Path, tmp_path: Path
) -> None:
    """The expensive path must not run again; a surviving sentinel proves it."""
    cfg = config_for(fixture_corpus_dir, tmp_path / "index")
    first = build_stack(cfg).index
    (first.path / SENTINEL).write_text("kilroy", encoding="utf-8")

    second = build_stack(cfg).index
    assert second.path == first.path
    assert second.meta.index_hash == first.meta.index_hash
    assert second.meta.created_at == first.meta.created_at
    assert (second.path / SENTINEL).is_file()


def test_forced_rebuild_replaces_the_index(fixture_corpus_dir: Path, tmp_path: Path) -> None:
    cfg = config_for(fixture_corpus_dir, tmp_path / "index")
    first = build_stack(cfg).index
    (first.path / SENTINEL).write_text("kilroy", encoding="utf-8")

    rebuilt = build_stack(cfg, force_index=True).index
    assert rebuilt.path == first.path
    assert not (rebuilt.path / SENTINEL).is_file()


def test_a_different_chunker_gets_a_different_index(
    fixture_corpus_dir: Path, tmp_path: Path
) -> None:
    root = tmp_path / "index"
    first = build_stack(config_for(fixture_corpus_dir, root, "chunker=fixed_token")).index
    second = build_stack(config_for(fixture_corpus_dir, root, "chunker=section_aware")).index
    assert first.meta.index_hash != second.meta.index_hash
    assert first.path != second.path


def test_a_different_embedder_gets_a_different_index(
    fixture_corpus_dir: Path, tmp_path: Path
) -> None:
    root = tmp_path / "index"
    first = build_stack(config_for(fixture_corpus_dir, root)).index
    second = build_stack(config_for(fixture_corpus_dir, root, "embedder.dim=128")).index
    assert first.meta.index_hash != second.meta.index_hash


def test_different_bm25_parameters_get_a_different_index(
    fixture_corpus_dir: Path, tmp_path: Path
) -> None:
    root = tmp_path / "index"
    first = build_stack(config_for(fixture_corpus_dir, root)).index
    second = build_stack(config_for(fixture_corpus_dir, root, "sparse.k1=2.0")).index
    assert first.meta.index_hash != second.meta.index_hash


def test_index_directory_is_named_by_its_hash(fixture_corpus_dir: Path, tmp_path: Path) -> None:
    index = build_stack(config_for(fixture_corpus_dir, tmp_path / "index")).index
    assert index.meta.index_hash.startswith(index.path.name)


def test_meta_records_what_it_was_built_from(fixture_corpus_dir: Path, tmp_path: Path) -> None:
    stack = build_stack(config_for(fixture_corpus_dir, tmp_path / "index"))
    meta = stack.index.meta
    assert meta.n_chunks == len(stack.chunks)
    assert meta.corpus_hash == stack.corpus.corpus_hash
    assert meta.embedder["name"] == "hashed"
    assert meta.chunker["name"] == stack.chunker.name
    assert meta.dim == stack.embedder.dim


def test_reloading_from_disk_matches(fixture_corpus_dir: Path, tmp_path: Path) -> None:
    stack = build_stack(config_for(fixture_corpus_dir, tmp_path / "index"))
    reloaded = load_index(stack.index.path)
    assert reloaded.order == stack.index.order
    assert reloaded.meta == stack.index.meta


def test_loading_a_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(IndexBuildError, match="no index at"):
        load_index(tmp_path / "absent")


def test_building_over_no_chunks_raises(fixture_corpus_dir: Path, tmp_path: Path) -> None:
    stack = build_stack(config_for(fixture_corpus_dir, tmp_path / "index"))
    with pytest.raises(IndexBuildError, match="zero chunks"):
        build_index(
            chunks=[],
            embedder=stack.embedder,
            chunker_fingerprint=stack.chunker.fingerprint(),
            sparse=stack.index.meta.sparse,
            corpus_hash=stack.corpus.corpus_hash,
            index_root=tmp_path / "index2",
        )
