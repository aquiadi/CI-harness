"""``evalgate config``: print the resolved config and the hashes it implies."""

from __future__ import annotations

from omegaconf import OmegaConf
from rich.console import Console

from evalgate.config import config_hash, load_config, resolve_path
from evalgate.corpus.manifest import load_manifest_if_present
from evalgate.hashing import short
from evalgate.prompts import load_prompt

console = Console()


def run(overrides: list[str]) -> int:
    """Print the composed config, its fingerprint, and referenced hashes."""
    cfg = load_config(overrides=overrides)
    console.print(OmegaConf.to_yaml(cfg, resolve=True).rstrip())

    console.print(f"\nconfig_hash: {config_hash(cfg)}")

    prompts_dir = resolve_path(cfg, "paths.prompts_dir")
    for key in ("generator.prompt", "judge.prompt"):
        value = OmegaConf.select(cfg, key)
        if isinstance(value, str):
            prompt = load_prompt(prompts_dir, value)
            console.print(f"prompt[{key}]: {prompt.name} {prompt.sha256}")

    manifest = load_manifest_if_present(resolve_path(cfg, "paths.corpus_manifest_path"))
    if manifest is None:
        console.print("corpus: no manifest yet (run `make corpus`)")
    else:
        console.print(
            f"corpus: {len(manifest.ok_entries)}/{len(manifest.entries)} documents, "
            f"corpus_hash {short(manifest.corpus_hash())}"
        )
    return 0
