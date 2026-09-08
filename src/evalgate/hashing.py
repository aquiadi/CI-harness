"""Content hashing.

Every artifact in this repo is identified by the SHA256 of its content: source
documents, chunk sets, indexes, prompts, configs and cache keys. Runs are only
comparable when the hashes they were produced under match, so hashing must be
stable across processes, machines and Python versions -- hence explicit UTF-8
encoding and sorted-key JSON everywhere below, and never ``hash()``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_READ_CHUNK_BYTES = 1 << 20
SHORT_LEN = 12


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the hex SHA256 of text, encoded as UTF-8."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Return the hex SHA256 of a file, read incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Serialise an object to JSON that is byte-identical for equal values.

    Keys are sorted, separators are fixed and non-ASCII characters are escaped,
    so the output does not depend on dict insertion order or locale.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def hash_obj(obj: Any) -> str:
    """Return the hex SHA256 of an object's canonical JSON form."""
    return sha256_text(canonical_json(obj))


def short(digest: str, length: int = SHORT_LEN) -> str:
    """Truncate a hex digest for use in directory names and report tables."""
    return digest[:length]


def combine(*digests: str) -> str:
    """Combine several digests into one, order-sensitively."""
    return sha256_text("\x1f".join(digests))
