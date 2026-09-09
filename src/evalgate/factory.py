"""Construction of config-selected components.

Hydra's ``_target_`` names the class; this module supplies the runtime
collaborators (tokenizer, index, embedder, reranker) that cannot live in YAML.
Each component receives only the collaborators its own signature declares, so
adding a strategy means adding a class and a YAML file -- there is no dispatch
table here to keep in sync.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

TARGET_KEY = "_target_"


def node_params(node: DictConfig, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Resolve a config node into constructor kwargs."""
    resolved = OmegaConf.to_container(node, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError(f"expected a config mapping, got {type(resolved).__name__}")
    skip = exclude | {TARGET_KEY}
    return {str(key): value for key, value in resolved.items() if str(key) not in skip}


def construct(
    node: DictConfig,
    exclude: frozenset[str] = frozenset(),
    **collaborators: Any,
) -> Any:
    """Instantiate ``node._target_`` with its YAML params plus what it accepts.

    Returns ``Any`` because the concrete class is chosen by config; callers
    narrow it to the Protocol they expect.
    """
    target = node.get(TARGET_KEY)
    if not isinstance(target, str):
        raise KeyError(f"config node has no {TARGET_KEY}: {node}")
    cls = get_class(target)
    accepted = {field.name for field in dataclasses.fields(cls)}
    supplied = {name: value for name, value in collaborators.items() if name in accepted}
    return cls(**node_params(node, exclude), **supplied)
