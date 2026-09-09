"""A local sqlite cache for model responses.

Keyed on (prompt hash, model, input hash) so that re-running a report, a sweep
or a labelling session costs nothing. The cache is not committed and is not the
cassette: it is a developer convenience, while cassettes are the reviewed,
committed record that CI replays.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from evalgate.corpus.manifest import utc_now_iso
from evalgate.models.base import ModelRequest, ModelResponse
from evalgate.models.serde import response_from_dict, response_to_dict

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    purpose     TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


class ResponseCache:
    """Content-addressed store of past model responses."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute(SCHEMA)
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, request: ModelRequest) -> ModelResponse | None:
        """Return a cached response, or None."""
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM responses WHERE key = ?", (request.cache_key(),)
            ).fetchone()
        if row is None:
            return None
        payload: dict[str, Any] = json.loads(row[0])
        return response_from_dict(payload, replayed=False, cache_hit=True)

    def put(self, request: ModelRequest, response: ModelResponse) -> None:
        """Store a response."""
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO responses "
                "(key, model, purpose, prompt_hash, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    request.cache_key(),
                    request.model,
                    request.purpose,
                    request.prompt_hash,
                    json.dumps(response_to_dict(response), sort_keys=True),
                    utc_now_iso(),
                ),
            )
            connection.commit()

    def size(self) -> int:
        """Number of cached responses."""
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) FROM responses").fetchone()
        return int(row[0])
