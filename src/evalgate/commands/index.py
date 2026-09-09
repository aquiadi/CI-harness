"""``evalgate index``: build or reuse the vector and sparse indexes."""

from __future__ import annotations

from rich.console import Console

from evalgate.config import load_config
from evalgate.hashing import short
from evalgate.pipeline import build_stack

console = Console()


def run(overrides: list[str]) -> int:
    """Build the index, reporting whether it was rebuilt or reused."""
    force = "index.force=true" in overrides
    stack = build_stack(
        load_config(overrides=[o for o in overrides if o != "index.force=true"]),
        force_index=force,
    )
    meta = stack.index.meta
    console.print(
        f"index {short(meta.index_hash)} at {stack.index.path}\n"
        f"  corpus     {stack.corpus.name} {short(meta.corpus_hash)} "
        f"({stack.corpus.size} documents)\n"
        f"  chunker    {meta.chunker['name']} -> {meta.n_chunks} chunks\n"
        f"  embedder   {meta.embedder['name']} / {meta.embedder['model']} dim {meta.dim}\n"
        f"  sparse     k1={meta.sparse.k1} b={meta.sparse.b}\n"
        f"  built at   {meta.created_at}"
    )
    return 0
