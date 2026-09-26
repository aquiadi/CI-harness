"""Judge scores on disk: one file per judge, with the stack that produced them.

Two defects shaped this module. First, every judge wrote to the same
`judge_scores.jsonl`, so running the rule-based baseline -- which the
calibration report itself recommended when no judgments existed -- silently
replaced the LLM judge's scores. Second, the scores carried no record of the
retrieval stack they were produced over, so the report described them with
whatever the current config happened to be. Rendered under the default
profile, it attributed an LLM judge's scores to the rule-based judge and a
bge index to a hashed one.

So each judge gets its own file, named after its model, and a sidecar written
at the same moment that says what produced it. A report describes judgments
by their own provenance, never by the configuration it is rendered under.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evalgate.evalsets.schemas import JudgeScore
from evalgate.evalsets.store import read_jsonl, write_jsonl

SCORES_SUFFIX = ".jsonl"
META_SUFFIX = ".meta.json"
HEURISTIC_PROVIDER = "heuristic"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class JudgmentsMeta(BaseModel):
    """What produced one judge's scores."""

    model_config = ConfigDict(extra="forbid")

    judge_model: str
    judge_provider: str
    judge_prompt: str
    judge_prompt_hash: str
    axes: list[str]
    scale_min: int
    scale_max: int
    answers_from: str
    # Whose answers were graded: "reference" for the eval set's own answers,
    # else a generator model. More than one when the same judge has graded
    # several generators, which is what the self-preference probe needs.
    generators: list[str]
    api_mode: str
    # Human-readable identity of the retrieval stack the judge saw context
    # from: corpus, chunker, embedder, retriever, index.
    stack: dict[str, str] = Field(default_factory=dict)
    # Set when this record was written after the fact rather than by
    # `make judge`, saying where its contents came from.
    note: str | None = None

    @property
    def is_heuristic(self) -> bool:
        """Whether this is a rule rather than a model."""
        return self.judge_provider == HEURISTIC_PROVIDER


@dataclass(frozen=True, slots=True)
class JudgmentSet:
    """One judge's scores and, when recorded, their provenance."""

    slug: str
    judgments: list[JudgeScore]
    meta: JudgmentsMeta | None

    @property
    def judge_model(self) -> str:
        """The judge, from its own records first."""
        if self.judgments:
            return self.judgments[0].judge_model
        return self.meta.judge_model if self.meta else self.slug

    @property
    def is_heuristic(self) -> bool:
        """Whether this set came from a rule; unknown provenance counts as a model."""
        return self.meta.is_heuristic if self.meta else False


def slug(judge_model: str) -> str:
    """A filename for a judge model id such as `qwen/qwen3.8-27b`."""
    return _UNSAFE.sub("_", judge_model).strip("_") or "judge"


def scores_path(directory: Path, judge_model: str) -> Path:
    """Where this judge's scores live."""
    return directory / f"{slug(judge_model)}{SCORES_SUFFIX}"


def write_judgments(directory: Path, meta: JudgmentsMeta, scores: list[JudgeScore]) -> Path:
    """Write one judge's scores and their provenance together."""
    path = scores_path(directory, meta.judge_model)
    write_jsonl(path, scores)
    meta_path = directory / f"{slug(meta.judge_model)}{META_SUFFIX}"
    meta_path.write_text(
        json.dumps(meta.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _load(directory: Path, name: str) -> JudgmentSet:
    meta_path = directory / f"{name}{META_SUFFIX}"
    meta = (
        JudgmentsMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        if meta_path.is_file()
        else None
    )
    judgments = read_jsonl(directory / f"{name}{SCORES_SUFFIX}", JudgeScore)
    return JudgmentSet(slug=name, judgments=judgments, meta=meta)


def load_judgment_set(directory: Path, judge_model: str) -> JudgmentSet | None:
    """One judge's scores, if it has any on disk. Other judges' files are not read."""
    if not scores_path(directory, judge_model).is_file():
        return None
    return _load(directory, slug(judge_model))


def load_judgment_sets(directory: Path) -> list[JudgmentSet]:
    """Every judge's scores on disk: models first, rules after, then by name.

    The order is what reports and the README lead with, so it is fixed: the
    LLM judge is the instrument under validation, and the rule is its baseline.
    """
    if not directory.is_dir():
        return []
    sets = [
        _load(directory, path.name.removesuffix(SCORES_SUFFIX))
        for path in sorted(directory.glob(f"*{SCORES_SUFFIX}"))
    ]
    return sorted(sets, key=lambda item: (item.is_heuristic, item.slug))
