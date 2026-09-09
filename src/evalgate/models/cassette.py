"""Record and replay of model calls.

This is the mechanism that lets pull-request CI run the full evaluation --
generation, judging, the quality gate -- deterministically, in seconds, for no
money. A cassette is one JSON file per call, holding the request that was made
and the response that came back, committed to the repository and reviewable in
a diff. In replay mode an unmatched call is an error: CI never silently falls
back to the network, and a change that alters a prompt or a model shows up as a
cassette miss rather than as a surprise invoice.

Cassettes are stored one file per call rather than one big file so that a diff
shows which calls changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalgate.corpus.manifest import utc_now_iso
from evalgate.hashing import short
from evalgate.models.base import ModelError, ModelRequest, ModelResponse
from evalgate.models.serde import response_from_dict, response_to_dict

CASSETTE_SUFFIX = ".json"


class CassetteMissError(ModelError):
    """Raised in replay mode when no cassette matches the request."""


class CassetteStore:
    """A directory of recorded calls."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _file(self, request: ModelRequest) -> Path:
        key = request.cache_key()
        return self.path / f"{request.purpose}-{short(key, 16)}{CASSETTE_SUFFIX}"

    def load(self, request: ModelRequest) -> ModelResponse | None:
        """Return the recorded response for this request, or None."""
        path = self._file(request)
        if not path.is_file():
            return None
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return response_from_dict(payload["response"], replayed=True, cache_hit=False)

    def save(self, request: ModelRequest, response: ModelResponse) -> Path:
        """Write a cassette for this call."""
        self.path.mkdir(parents=True, exist_ok=True)
        path = self._file(request)
        payload = {
            "recorded_at": utc_now_iso(),
            "key": request.cache_key(),
            "request": {
                "model": request.model,
                "purpose": request.purpose,
                "prompt_hash": request.prompt_hash,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "system": request.system,
                "prompt": request.prompt,
                "tool": request.tool.to_api() if request.tool else None,
            },
            "response": response_to_dict(response),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def count(self) -> int:
        """Number of cassettes on disk."""
        return len(list(self.path.glob(f"*{CASSETTE_SUFFIX}"))) if self.path.is_dir() else 0
