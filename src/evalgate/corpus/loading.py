"""Turning the pinned corpus on disk into documents.

Two sources are supported and both produce the same ``Document`` objects: the
downloaded PDFs listed in the manifest, and a local directory of text files.
The second exists so the pipeline can run where the source documents are not
reachable; a corpus loaded that way carries its own hash and its own name, so
nothing produced from it can be confused with a run over the real instruments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evalgate.corpus.manifest import CorpusManifest, load_manifest_if_present
from evalgate.hashing import hash_obj, sha256_file
from evalgate.ingest.documents import Document, IngestError, load_document

LOCAL_SUFFIXES = (".md", ".txt")


@dataclass(frozen=True, slots=True)
class LoadedCorpus:
    """The documents a run was built from, and their collective identity."""

    name: str
    documents: list[Document]
    corpus_hash: str

    @property
    def size(self) -> int:
        """Number of documents."""
        return len(self.documents)


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return fallback


def load_local_corpus(name: str, directory: Path) -> LoadedCorpus:
    """Load every text document in a directory, ordered by filename."""
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in LOCAL_SUFFIXES
    )
    if not paths:
        raise IngestError(f"no {'/'.join(LOCAL_SUFFIXES)} documents under {directory}")

    documents: list[Document] = []
    digests: list[tuple[str, str]] = []
    for path in paths:
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        documents.append(load_document(path, doc_id, _title_from_text(text, doc_id)))
        digests.append((doc_id, sha256_file(path)))
    return LoadedCorpus(name=name, documents=documents, corpus_hash=hash_obj(sorted(digests)))


def load_manifest_corpus(name: str, raw_dir: Path, manifest: CorpusManifest) -> LoadedCorpus:
    """Load the documents the manifest reports as present."""
    documents: list[Document] = []
    for entry in manifest.ok_entries:
        path = raw_dir / entry.filename
        if not path.is_file():
            raise IngestError(
                f"manifest lists {entry.id} as present but {path} is missing; run `make corpus`"
            )
        documents.append(load_document(path, entry.id, entry.title))
    if not documents:
        raise IngestError(
            "the manifest records no successfully fetched documents. "
            "Run `make corpus` on a network that can reach the source URLs, "
            "or point corpus.local_dir at a local document set."
        )
    return LoadedCorpus(name=name, documents=documents, corpus_hash=manifest.corpus_hash())


def load_corpus(
    name: str,
    raw_dir: Path,
    manifest_path: Path,
    local_dir: Path | None = None,
) -> LoadedCorpus:
    """Load the corpus from a local directory if configured, else the manifest."""
    if local_dir is not None:
        return load_local_corpus(name, local_dir)
    manifest = load_manifest_if_present(manifest_path)
    if manifest is None:
        raise IngestError(f"no corpus manifest at {manifest_path}; run `make corpus`")
    return load_manifest_corpus(name, raw_dir, manifest)
