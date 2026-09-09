"""Assembling the retrieval stack from config.

One place where corpus, tokenizer, chunker, embedder, index and retriever are
wired together, so that the CLI, the eval loop, the ablation sweep and the
FastAPI layer all run the identical path. A second wiring site is how a serving
stack quietly drifts from the one that was measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

from evalgate.chunking.base import Chunk, Chunker
from evalgate.config import SparseConfig, resolve_path, typed_node
from evalgate.corpus.loading import LoadedCorpus, load_corpus
from evalgate.embeddings.base import Embedder
from evalgate.factory import construct
from evalgate.ingest.tokens import TokenEstimator
from evalgate.retrieval.base import Retriever
from evalgate.retrieval.hybrid import Reranker
from evalgate.retrieval.index import ChunkIndex, SparseParams, build_index

RERANK_KEY = "rerank"


@dataclass(frozen=True, slots=True)
class RetrievalStack:
    """Everything needed to answer a query, plus the identity of each part."""

    corpus: LoadedCorpus
    tokenizer: TokenEstimator
    chunker: Chunker
    embedder: Embedder
    chunks: list[Chunk]
    index: ChunkIndex
    retriever: Retriever


def build_tokenizer(cfg: DictConfig) -> TokenEstimator:
    """Instantiate the configured token estimator."""
    return cast(TokenEstimator, construct(cfg.tokenizer))


def build_chunker(cfg: DictConfig, tokenizer: TokenEstimator) -> Chunker:
    """Instantiate the configured chunker."""
    return cast(Chunker, construct(cfg.chunker, tokenizer=tokenizer))


def build_embedder(cfg: DictConfig) -> Embedder:
    """Instantiate the configured embedding backend."""
    return cast(Embedder, construct(cfg.embedder))


def load_documents(cfg: DictConfig) -> LoadedCorpus:
    """Load the configured corpus from disk."""
    local = cfg.corpus.get("local_dir")
    local_dir: Path | None = None
    if isinstance(local, str) and local:
        local_dir = resolve_path(cfg, "corpus.local_dir")
    return load_corpus(
        name=str(cfg.corpus.name),
        raw_dir=resolve_path(cfg, "paths.corpus_raw_dir"),
        manifest_path=resolve_path(cfg, "paths.corpus_manifest_path"),
        local_dir=local_dir,
    )


def chunk_corpus(corpus: LoadedCorpus, chunker: Chunker) -> list[Chunk]:
    """Chunk every document, in document order."""
    chunks: list[Chunk] = []
    for document in corpus.documents:
        chunks.extend(chunker.chunk(document))
    return chunks


def sparse_params(cfg: DictConfig) -> SparseParams:
    """The BM25 parameters the index is built with."""
    sparse = typed_node(cfg, "sparse", SparseConfig)
    return SparseParams(
        name=sparse.name,
        k1=sparse.k1,
        b=sparse.b,
        stopwords=sparse.stopwords,
        stemmer=sparse.stemmer,
    )


def build_reranker(cfg: DictConfig) -> tuple[Reranker | None, int]:
    """Instantiate the reranker if the retriever config enables one."""
    node = cfg.retriever.get(RERANK_KEY)
    if node is None or not bool(node.get("enabled", False)):
        return None, 0
    reranker = cast(Reranker, construct(node, exclude=frozenset({"enabled"})))
    return reranker, int(node.get("top_n", 0))


def build_retriever(cfg: DictConfig, index: ChunkIndex, embedder: Embedder) -> Retriever:
    """Instantiate the configured retriever over an existing index."""
    reranker, top_n = build_reranker(cfg)
    return cast(
        Retriever,
        construct(
            cfg.retriever,
            exclude=frozenset({RERANK_KEY}),
            index=index,
            embedder=embedder,
            reranker=reranker,
            rerank_top_n=top_n,
        ),
    )


def build_stack(cfg: DictConfig, force_index: bool = False) -> RetrievalStack:
    """Load the corpus, chunk it, build or reuse the index, wire the retriever."""
    tokenizer = build_tokenizer(cfg)
    chunker = build_chunker(cfg, tokenizer)
    embedder = build_embedder(cfg)
    corpus = load_documents(cfg)
    chunks = chunk_corpus(corpus, chunker)
    index = build_index(
        chunks=chunks,
        embedder=embedder,
        chunker_fingerprint=chunker.fingerprint(),
        sparse=sparse_params(cfg),
        corpus_hash=corpus.corpus_hash,
        index_root=resolve_path(cfg, "paths.index_dir"),
        force=force_index,
    )
    retriever = build_retriever(cfg, index, embedder)
    return RetrievalStack(
        corpus=corpus,
        tokenizer=tokenizer,
        chunker=chunker,
        embedder=embedder,
        chunks=chunks,
        index=index,
        retriever=retriever,
    )


def stack_fingerprint(stack: RetrievalStack) -> dict[str, object]:
    """The identity of every component in the stack, for the run record."""
    return {
        "corpus": {"name": stack.corpus.name, "hash": stack.corpus.corpus_hash},
        "tokenizer": stack.tokenizer.fingerprint(),
        "chunker": stack.chunker.fingerprint(),
        "embedder": stack.embedder.fingerprint(),
        "retriever": stack.retriever.fingerprint(),
        "index": {"hash": stack.index.meta.index_hash, "n_chunks": stack.index.meta.n_chunks},
    }


def resolved(cfg: DictConfig, key: str) -> object:
    """Resolve one config key to a plain Python value."""
    return OmegaConf.to_container(cfg.get(key), resolve=True)
