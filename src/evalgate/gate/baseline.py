"""The frozen baseline.

`baseline.json` records what the winning configuration measured, together with
the hashes of everything upstream of those numbers: the corpus, the prompts and
the eval sets. Freezing is deliberate and visible in a diff, because moving the
baseline is how a quality regression becomes permanent.

Note which hashes are *not* here as a comparability requirement: the config
hash and the index hash. A pull request that changes the retriever, the chunker
or k changes both, and that is exactly the case the gate exists to judge. What
must not change silently is the ground the comparison stands on -- the same
documents, the same prompts, the same questions, and the same judge. The judge
is the instrument, not the system under test: a composite scored by a
different judge is in different units, and a drop or a gain against the
baseline would describe the ruler rather than the change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evalgate.errors import EvalgateError
from evalgate.hashing import hash_obj
from evalgate.runs.store import RunMeta

# 2 added `items`: one record per question, so a gate run can say which
# questions moved. A version-1 baseline still loads, with no items.
SCHEMA_VERSION = 2


class BaselineError(EvalgateError):
    """Raised when a baseline is missing or cannot be compared against."""


class Baseline(BaseModel):
    """A frozen measurement, and the ground it was measured on."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: str
    frozen_at: str
    config_hash: str
    corpus_name: str
    corpus_hash: str
    index_hash: str
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    evalset_hashes: dict[str, str] = Field(default_factory=dict)
    api_mode: str
    replayed: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    fingerprint: dict[str, Any] = Field(default_factory=dict)
    items: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    # How the frozen run timed latency; empty for runs timed once, cold.
    timing: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None

    def mismatches(self, meta: RunMeta) -> list[str]:
        """What differs between this baseline's ground and a run's.

        Returns human-readable descriptions, empty when the two are comparable.
        """
        differences: list[str] = []
        judge_before = self.fingerprint.get("judge")
        judge_after = meta.fingerprint.get("judge")
        if judge_before != judge_after:
            differences.append(
                f"judge: baseline {_judge_name(judge_before)}, run {_judge_name(judge_after)}"
            )
        if meta.corpus_hash != self.corpus_hash:
            differences.append(
                f"corpus hash: baseline {self.corpus_hash[:12]}, run {meta.corpus_hash[:12]}"
            )
        if meta.prompt_hashes != self.prompt_hashes:
            for name in sorted(set(self.prompt_hashes) | set(meta.prompt_hashes)):
                before = self.prompt_hashes.get(name, "absent")
                after = meta.prompt_hashes.get(name, "absent")
                if before != after:
                    differences.append(f"prompt {name}: baseline {before[:12]}, run {after[:12]}")
        if meta.evalset_hashes != self.evalset_hashes:
            for name in sorted(set(self.evalset_hashes) | set(meta.evalset_hashes)):
                before = self.evalset_hashes.get(name, "absent")
                after = meta.evalset_hashes.get(name, "absent")
                if before != after:
                    differences.append(f"eval set {name}: baseline {before[:12]}, run {after[:12]}")
        return differences


def _judge_name(node: Any) -> str:
    """A judge's model and a short hash of its full fingerprint.

    The hash distinguishes two configurations of one model -- a different
    prompt or reasoning effort is a different instrument too.
    """
    if not isinstance(node, dict):
        return "none"
    return f"{node.get('model') or node.get('name') or '?'} ({hash_obj(node)[:8]})"


def freeze(
    meta: RunMeta,
    metrics: dict[str, Any],
    frozen_at: str,
    notes: str | None = None,
    items: dict[str, list[dict[str, Any]]] | None = None,
) -> Baseline:
    """Build a baseline from a run."""
    return Baseline(
        run_id=meta.run_id,
        frozen_at=frozen_at,
        config_hash=meta.config_hash,
        corpus_name=meta.corpus_name,
        corpus_hash=meta.corpus_hash,
        index_hash=meta.index_hash,
        prompt_hashes=dict(meta.prompt_hashes),
        evalset_hashes=dict(meta.evalset_hashes),
        api_mode=meta.api_mode,
        replayed=meta.replayed,
        metrics=dict(metrics),
        fingerprint=dict(meta.fingerprint),
        items=dict(items or {}),
        timing=dict(meta.timing),
        notes=notes,
    )


def write_baseline(path: Path, baseline: Baseline) -> None:
    """Write the baseline as stable, reviewable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = baseline.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_baseline(path: Path) -> Baseline:
    """Read the baseline, failing loudly when it is absent.

    A missing baseline is a gate failure, never a pass. "Nothing to compare
    against" is the state in which a regression is invisible.
    """
    if not path.is_file():
        raise BaselineError(
            f"no baseline at {path}. The gate cannot pass without something to compare "
            "against; create one with `make freeze` and commit it."
        )
    return Baseline.model_validate_json(path.read_text(encoding="utf-8"))
