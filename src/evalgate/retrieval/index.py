"""The chunk index: dense vectors in LanceDB, sparse postings in bm25s.

Two properties matter more than speed here.

*Determinism.* No ANN index is created. LanceDB does an exact brute-force scan,
which for a corpus of this size costs milliseconds and removes an entire class
of "the numbers moved and nobody changed anything" bug that approximate indexes
introduce.

*Content addressing.* The index directory is named by a hash of everything that
determines its contents -- corpus, chunker, embedder, BM25 parameters. Building
an index whose hash already exists on disk is a no-op, so ablations that share a
chunker share an index instead of rebuilding it once per configuration.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from evalgate.chunking.base import Chunk
from evalgate.corpus.manifest import utc_now_iso
from evalgate.embeddings.base import Embedder
from evalgate.errors import EvalgateError
from evalgate.hashing import hash_obj, short
from evalgate.retrieval.base import rank_scores

CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"
VECTORS_DIR = "lancedb"
SPARSE_DIR = "bm25"
TABLE_NAME = "chunks"


class IndexBuildError(EvalgateError):
    """Raised when an index cannot be built or loaded."""


class SparseParams(BaseModel):
    """BM25 index-time parameters, recorded so the index can be identified."""

    model_config = ConfigDict(extra="forbid")

    name: str
    k1: float
    b: float
    stopwords: str
    stemmer: str


class IndexMeta(BaseModel):
    """What an index was built from."""

    model_config = ConfigDict(extra="forbid")

    index_hash: str
    corpus_hash: str
    n_chunks: int
    dim: int
    embedder: dict[str, Any]
    chunker: dict[str, Any]
    sparse: SparseParams
    created_at: str


def index_hash(
    corpus_hash: str,
    chunker: dict[str, Any],
    embedder: dict[str, Any],
    sparse: SparseParams,
) -> str:
    """Hash of everything that determines the index contents."""
    return hash_obj(
        {
            "corpus": corpus_hash,
            "chunker": chunker,
            "embedder": embedder,
            "sparse": sparse.model_dump(),
        }
    )


def _tokenize(texts: list[str], sparse: SparseParams) -> Any:
    import bm25s

    stemmer = None if sparse.stemmer == "none" else sparse.stemmer
    return bm25s.tokenize(
        texts,
        stopwords=sparse.stopwords,
        stemmer=stemmer,
        show_progress=False,
    )


@dataclass(slots=True)
class ChunkIndex:
    """A built index over one chunk set."""

    path: Path
    meta: IndexMeta
    chunks: dict[str, Chunk]
    order: list[str]
    _table: Any
    _bm25: Any

    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self.order)

    def get(self, chunk_id: str) -> Chunk:
        """Look up one chunk."""
        return self.chunks[chunk_id]

    def dense_search(self, vector: np.ndarray[Any, Any], k: int) -> list[tuple[str, float]]:
        """Exact cosine search. Returns (chunk_id, similarity) pairs.

        A query whose embedding has zero norm (no indexable content: "?", an
        emoji, an empty string) has no direction to compare against, and
        LanceDB drops the undefined distances, returning nothing. Rather than
        let one retriever silently return fewer results than the others for
        such a query, score everything zero and fall through to the shared
        tie-break -- which is what BM25 already does for an out-of-vocabulary
        query. Real eval questions have signal; this only makes the contract
        uniform for the ones that do not.
        """
        limit = min(max(1, k), self.size)
        query = np.asarray(vector, dtype=np.float32)
        if not np.any(query):
            return rank_scores([(chunk_id, 0.0) for chunk_id in self.order], limit)
        rows = self._table.search(query).metric("cosine").limit(limit).to_list()
        # LanceDB reports cosine distance; similarity is what the rest of the
        # pipeline compares, and it must be non-increasing down the list.
        scored = [(str(row["id"]), 1.0 - float(row["_distance"])) for row in rows]
        return rank_scores(scored, limit)

    def sparse_search(self, query: str, k: int) -> list[tuple[str, float]]:
        """BM25 search. Returns (chunk_id, score) pairs."""
        limit = min(max(1, k), self.size)
        tokens = _tokenize([query], self.meta.sparse)
        indices, scores = self._bm25.retrieve(tokens, k=limit, show_progress=False)
        scored = [
            (self.order[int(doc)], float(score))
            for doc, score in zip(indices[0], scores[0], strict=True)
        ]
        return rank_scores(scored, limit)


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(_chunk_dict(chunk)) + "\n")


def _chunk_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "doc_id": chunk.doc_id,
        "doc_title": chunk.doc_title,
        "ordinal": chunk.ordinal,
        "text": chunk.text,
        "n_tokens": chunk.n_tokens,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "section": chunk.section,
    }


def _read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def build_index(
    chunks: list[Chunk],
    embedder: Embedder,
    chunker_fingerprint: dict[str, Any],
    sparse: SparseParams,
    corpus_hash: str,
    index_root: Path,
    force: bool = False,
) -> ChunkIndex:
    """Build (or reuse) the index for this chunk set.

    Rebuilding an unchanged corpus is a no-op: the directory name is the
    content hash, and an existing directory with a valid meta file is loaded
    rather than recomputed.
    """
    if not chunks:
        raise IndexBuildError("refusing to build an index over zero chunks")

    digest = index_hash(corpus_hash, chunker_fingerprint, embedder.fingerprint(), sparse)
    path = index_root / short(digest)
    if path.is_dir() and (path / META_FILE).is_file() and not force:
        existing = load_index(path)
        if existing.meta.index_hash == digest:
            return existing
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    ordered = sorted(chunks, key=lambda chunk: chunk.id)
    vectors = embedder.embed_documents([chunk.text for chunk in ordered])
    expected = (len(ordered), embedder.dim)
    if vectors.shape != expected:
        raise IndexBuildError(f"embedder returned {vectors.shape}, expected {expected}")

    import lancedb

    db = lancedb.connect(str(path / VECTORS_DIR))
    db.create_table(
        TABLE_NAME,
        data=[
            {"id": chunk.id, "text": chunk.text, "vector": vector.tolist()}
            for chunk, vector in zip(ordered, vectors, strict=True)
        ],
        mode="overwrite",
    )

    import bm25s

    bm25 = bm25s.BM25(k1=sparse.k1, b=sparse.b)
    bm25.index(_tokenize([chunk.text for chunk in ordered], sparse), show_progress=False)
    bm25.save(str(path / SPARSE_DIR))

    _write_chunks(path / CHUNKS_FILE, ordered)
    meta = IndexMeta(
        index_hash=digest,
        corpus_hash=corpus_hash,
        n_chunks=len(ordered),
        dim=embedder.dim,
        embedder=embedder.fingerprint(),
        chunker=chunker_fingerprint,
        sparse=sparse,
        created_at=utc_now_iso(),
    )
    (path / META_FILE).write_text(
        json.dumps(meta.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return load_index(path)


def load_index(path: Path) -> ChunkIndex:
    """Load a previously built index."""
    meta_path = path / META_FILE
    if not meta_path.is_file():
        raise IndexBuildError(f"no index at {path} (run `make index`)")
    meta = IndexMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    chunks = _read_chunks(path / CHUNKS_FILE)

    import bm25s
    import lancedb

    table = lancedb.connect(str(path / VECTORS_DIR)).open_table(TABLE_NAME)
    bm25 = bm25s.BM25.load(str(path / SPARSE_DIR))
    return ChunkIndex(
        path=path,
        meta=meta,
        chunks={chunk.id: chunk for chunk in chunks},
        order=[chunk.id for chunk in chunks],
        _table=table,
        _bm25=bm25,
    )
