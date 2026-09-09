"""JSONL storage for eval sets and labels.

Append-only where it can be: the labelling CLI writes each rating the moment it
is given and flushes to disk, so a session that ends in a Ctrl-C, a closed
laptop or a dropped connection loses nothing. Rewrites go through a temporary
file and an atomic rename, so an interrupted rewrite cannot truncate a file
that took a person an hour to produce.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError

from evalgate.errors import EvalgateError


class EvalStoreError(EvalgateError):
    """Raised when an eval file cannot be read."""


def read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    """Read and validate every record, reporting the offending line number."""
    if not path.is_file():
        return []
    records: list[T] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError as exc:
                raise EvalStoreError(f"{path}:{number}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    """Rewrite a file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, record: BaseModel) -> None:
    """Append one record and force it to disk before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def index_by[T: BaseModel](records: Sequence[T], key: str) -> dict[str, T]:
    """Index records by a string attribute, last write winning."""
    return {str(getattr(record, key)): record for record in records}
