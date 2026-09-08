"""The corpus manifest.

The source PDFs are not committed -- they are large, and redistributing them is
not our call. What is committed is this manifest: for every pinned URL, the
SHA256 of the bytes that were actually downloaded. That makes the corpus
reproducible (anyone can re-fetch and verify) and makes silent upstream edits
visible as a hash change rather than as unexplained movement in eval numbers.

A URL that cannot be fetched is recorded with its failure and the run
continues. A partial corpus is a measurable condition, not a crash.
"""

from __future__ import annotations

import datetime as dt
import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evalgate.hashing import hash_obj

SCHEMA_VERSION = 1


class FetchStatus(StrEnum):
    """Outcome of fetching one source document."""

    OK = "ok"
    MISSING = "missing"
    ERROR = "error"


class ManifestEntry(BaseModel):
    """One pinned source document and what happened when we asked for it."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    url: str
    filename: str
    kind: str
    required: bool
    status: FetchStatus
    sha256: str | None = None
    bytes: int | None = None
    http_status: int | None = None
    content_type: str | None = None
    error: str | None = None
    fetched_at: str | None = None


class CorpusManifest(BaseModel):
    """The full pinned document set."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    corpus: str
    generated_at: str
    entries: list[ManifestEntry] = Field(default_factory=list)

    @property
    def ok_entries(self) -> list[ManifestEntry]:
        """Entries whose bytes are on disk and hashed."""
        return [e for e in self.entries if e.status is FetchStatus.OK]

    @property
    def failed_entries(self) -> list[ManifestEntry]:
        """Entries that could not be fetched."""
        return [e for e in self.entries if e.status is not FetchStatus.OK]

    def corpus_hash(self) -> str:
        """Identity of the document set actually present.

        Only successful entries contribute, so a corpus that later gains a
        previously-unreachable document hashes differently -- which is correct:
        runs across the two are not comparable.
        """
        payload = sorted((e.id, e.sha256) for e in self.ok_entries)
        return hash_obj(payload)

    def entry(self, source_id: str) -> ManifestEntry | None:
        """Look up one entry by source id."""
        return next((e for e in self.entries if e.id == source_id), None)


def utc_now_iso() -> str:
    """Timestamp used throughout the repo: UTC, second precision, Z-suffixed."""
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_manifest(path: Path, manifest: CorpusManifest) -> None:
    """Write the manifest as stable, diffable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> CorpusManifest:
    """Read and validate a manifest."""
    return CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_manifest_if_present(path: Path) -> CorpusManifest | None:
    """Read a manifest, returning None when it does not exist yet."""
    if not path.is_file():
        return None
    return read_manifest(path)
