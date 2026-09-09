"""Run artifacts: one immutable directory per run.

    runs/<timestamp>-<confighash>/
        meta.json      what produced this run: config, prompt and corpus hashes
        config.yaml    the fully resolved config, for reproduction
        rows.parquet   one row per eval question
        metrics.json   the aggregates the reports and the gate read

The directory is written once. Writing into a directory that already exists is
refused rather than merged: a run whose contents can change is not evidence,
and the gate's whole job is to compare evidence.

Parquet rather than CSV because the row schema has lists (retrieved chunk ids)
and floats whose precision matters, and because pandas reads a directory of
them without a server (docs/DECISIONS.md D-0007).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from evalgate.corpus.manifest import utc_now_iso
from evalgate.errors import EvalgateError
from evalgate.hashing import short

META_FILE = "meta.json"
CONFIG_FILE = "config.yaml"
ROWS_FILE = "rows.parquet"
METRICS_FILE = "metrics.json"
RUN_ID_FORMAT = "{timestamp}-{config_hash}"


class RunExistsError(EvalgateError):
    """Raised when a run directory would be overwritten."""


class RunNotFoundError(EvalgateError):
    """Raised when a run directory cannot be read."""


class RunMeta(BaseModel):
    """Everything needed to say whether two runs are comparable."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: str
    config_hash: str
    corpus_name: str
    corpus_hash: str
    index_hash: str
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    evalset_hashes: dict[str, str] = Field(default_factory=dict)
    api_mode: str
    replayed: bool
    fingerprint: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None

    def comparable_to(self, other: RunMeta) -> bool:
        """Two runs are comparable only if everything upstream of the numbers matches.

        Deliberately strict. A footnote saying "these used different prompts"
        is not a substitute for refusing to put them in one table.
        """
        return (
            self.corpus_hash == other.corpus_hash
            and self.prompt_hashes == other.prompt_hashes
            and self.index_hash == other.index_hash
            and self.evalset_hashes == other.evalset_hashes
        )


def make_run_id(config_hash: str, timestamp: str | None = None) -> str:
    """Directory name for a run: sortable by time, identified by config."""
    stamp = (timestamp or utc_now_iso()).replace(":", "").replace("-", "")
    return RUN_ID_FORMAT.format(timestamp=stamp, config_hash=short(config_hash))


def write_run(
    runs_root: Path,
    meta: RunMeta,
    rows: pd.DataFrame,
    metrics: dict[str, Any],
    cfg: DictConfig | None = None,
) -> Path:
    """Write a run directory, refusing to touch one that already exists."""
    path = runs_root / meta.run_id
    if path.exists():
        raise RunExistsError(f"run directory already exists and runs are immutable: {path}")
    path.mkdir(parents=True)

    rows.to_parquet(path / ROWS_FILE, index=False)
    (path / META_FILE).write_text(
        json.dumps(meta.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (path / METRICS_FILE).write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    if cfg is not None:
        (path / CONFIG_FILE).write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return path


def load_meta(path: Path) -> RunMeta:
    """Read one run's metadata."""
    meta_path = path / META_FILE
    if not meta_path.is_file():
        raise RunNotFoundError(f"no run at {path}")
    return RunMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> pd.DataFrame:
    """Read one run's per-question rows."""
    rows_path = path / ROWS_FILE
    if not rows_path.is_file():
        raise RunNotFoundError(f"no rows at {rows_path}")
    return pd.read_parquet(rows_path)


def load_metrics(path: Path) -> dict[str, Any]:
    """Read one run's aggregate metrics."""
    metrics_path = path / METRICS_FILE
    if not metrics_path.is_file():
        raise RunNotFoundError(f"no metrics at {metrics_path}")
    payload: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payload


def list_runs(runs_root: Path) -> list[Path]:
    """Every run directory, oldest first."""
    if not runs_root.is_dir():
        return []
    return sorted(path for path in runs_root.iterdir() if (path / META_FILE).is_file())


def resolve_run(runs_root: Path, run_id: str | None) -> Path:
    """Locate a run by id, defaulting to the most recent."""
    runs = list_runs(runs_root)
    if not runs:
        raise RunNotFoundError(f"no runs under {runs_root}")
    if run_id is None:
        return runs[-1]
    for path in runs:
        if path.name == run_id:
            return path
    raise RunNotFoundError(f"no run {run_id!r} under {runs_root}")


def load_run_answers(runs_root: Path, run_id: str | None) -> list[dict[str, Any]]:
    """The answers a run produced, as plain records."""
    frame = load_rows(resolve_run(runs_root, run_id))
    return [{str(key): value for key, value in row.items()} for row in frame.to_dict("records")]
