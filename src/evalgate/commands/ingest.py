"""``evalgate ingest``: corpus -> chunks, with the distribution printed."""

from __future__ import annotations

import json
import statistics

from rich.console import Console
from rich.table import Table

from evalgate.config import load_config, resolve_path
from evalgate.hashing import short
from evalgate.pipeline import build_chunker, build_tokenizer, chunk_corpus, load_documents

console = Console()


def run(overrides: list[str]) -> int:
    """Chunk the corpus and write the chunk set to the processed directory."""
    cfg = load_config(overrides=overrides)
    tokenizer = build_tokenizer(cfg)
    chunker = build_chunker(cfg, tokenizer)
    corpus = load_documents(cfg)
    chunks = chunk_corpus(corpus, chunker)

    out_dir = resolve_path(cfg, "paths.corpus_processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{corpus.name}-{cfg.chunker.name}-{short(corpus.corpus_hash)}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "id": chunk.id,
                        "doc_id": chunk.doc_id,
                        "doc_title": chunk.doc_title,
                        "ordinal": chunk.ordinal,
                        "n_tokens": chunk.n_tokens,
                        "section": chunk.section,
                        "text": chunk.text,
                    }
                )
                + "\n"
            )

    table = Table(title=f"{corpus.name} / {cfg.chunker.name}", show_edge=False)
    table.add_column("document")
    table.add_column("chunks", justify="right")
    table.add_column("tokens", justify="right")
    for document in corpus.documents:
        owned = [chunk for chunk in chunks if chunk.doc_id == document.id]
        table.add_row(document.id, str(len(owned)), str(sum(c.n_tokens for c in owned)))
    console.print(table)

    sizes = [chunk.n_tokens for chunk in chunks]
    console.print(
        f"{len(chunks)} chunks, tokens min/median/max "
        f"{min(sizes)}/{int(statistics.median(sizes))}/{max(sizes)}, "
        f"corpus_hash {short(corpus.corpus_hash)}"
    )
    console.print(f"written to {out_path}")
    return 0
