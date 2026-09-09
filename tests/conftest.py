"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from omegaconf import DictConfig

from evalgate.config import load_config
from evalgate.pipeline import RetrievalStack, build_stack
from evalgate.rootdir import find_repo_root


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return find_repo_root()


@pytest.fixture
def cfg() -> DictConfig:
    """The default composed config."""
    return load_config()


@pytest.fixture(scope="session")
def fixture_corpus_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "corpus"


@pytest.fixture
def fixture_cfg(fixture_corpus_dir: Path, tmp_path: Path) -> DictConfig:
    """A config over the fixture corpus, with no model downloads and no network.

    Uses the deterministic embedder and a per-test index directory so that
    tests never touch the developer's real index.
    """
    return load_config(
        overrides=[
            "corpus.name=fixture",
            f"corpus.local_dir={fixture_corpus_dir}",
            "embedder=hashed",
            f"paths.index_dir={tmp_path / 'index'}",
        ]
    )


@pytest.fixture
def fixture_stack(fixture_cfg: DictConfig) -> Iterator[RetrievalStack]:
    yield build_stack(fixture_cfg)
