"""Repository root discovery.

This is the one bootstrap path in the codebase. Everything else -- data
directories, index locations, prompt files, thresholds -- comes from hydra
config. To find the config in the first place we have to start somewhere, so:
``EVALGATE_ROOT`` if set, otherwise the nearest ancestor of the working
directory that contains ``configs/config.yaml``, otherwise the package parent.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

ROOT_ENV_VAR = "EVALGATE_ROOT"
CONFIG_DIR_NAME = "configs"
CONFIG_SENTINEL = Path(CONFIG_DIR_NAME) / "config.yaml"


@lru_cache(maxsize=1)
def find_repo_root() -> Path:
    """Return the repository root directory."""
    override = os.environ.get(ROOT_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / CONFIG_SENTINEL).is_file():
            return candidate

    # Installed as a wheel with no checkout in sight: fall back to the package's
    # grandparent so that `src/evalgate/rootdir.py` resolves to the repo root.
    package_root = Path(__file__).resolve().parent.parent.parent
    return package_root


def config_dir() -> Path:
    """Return the directory holding the hydra config tree."""
    return find_repo_root() / CONFIG_DIR_NAME
