"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from evalgate.config import load_config
from evalgate.rootdir import find_repo_root


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return find_repo_root()


@pytest.fixture
def cfg() -> DictConfig:
    """The default composed config."""
    return load_config()
