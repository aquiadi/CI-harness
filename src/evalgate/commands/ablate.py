"""``evalgate ablate``: sweep the config matrix, one immutable run per cell.

Every cell of {chunker} x {retriever} x {k} x {rerank} that is meaningful gets
its own run directory under `runs/`, named by timestamp and config hash. Cells
that are not meaningful are skipped and reported rather than silently dropped:
reranking a lexical-only retriever is not a configuration, it is a mistake, and
a sweep that quietly ignores it leaves the reader wondering what happened to
the missing rows.

Indexes are shared across cells that agree on corpus, chunker and embedder, so
a nine-cell sweep over three chunkers builds three indexes, not nine.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from omegaconf import DictConfig
from rich.console import Console

from evalgate.config import AblationConfig, load_config, resolve_path, typed_node
from evalgate.embeddings.local import BackendUnavailableError
from evalgate.evaluation.runner import evaluate
from evalgate.generation.factory import build_generator
from evalgate.judging.factory import build_judge
from evalgate.models.client import build_model_client
from evalgate.pipeline import build_stack
from evalgate.prompts import load_prompt
from evalgate.runs.store import RunExistsError, write_run

console = Console()

RERANKING_RETRIEVERS = frozenset({"hybrid"})
RERANK_SUFFIX = "_rerank"


@dataclass(frozen=True, slots=True)
class Cell:
    """One point in the sweep."""

    chunker: str
    retriever: str
    k: int
    rerank: bool

    @property
    def overrides(self) -> list[str]:
        """Hydra overrides that select this cell."""
        retriever = f"{self.retriever}{RERANK_SUFFIX}" if self.rerank else self.retriever
        return [f"chunker={self.chunker}", f"retriever={retriever}", f"retriever.k={self.k}"]

    def __str__(self) -> str:
        """Compact label used in logs and the report."""
        return f"{self.chunker}/{self.retriever}{'+rerank' if self.rerank else ''}/k={self.k}"


def expand(matrix: AblationConfig) -> tuple[list[Cell], list[tuple[Cell, str]]]:
    """Build the sweep, separating meaningful cells from skipped ones."""
    cells: list[Cell] = []
    skipped: list[tuple[Cell, str]] = []
    for chunker, retriever, k, rerank in itertools.product(
        matrix.chunkers, matrix.retrievers, matrix.k_values, matrix.rerank
    ):
        cell = Cell(chunker=chunker, retriever=retriever, k=int(k), rerank=bool(rerank))
        if rerank and retriever not in RERANKING_RETRIEVERS:
            skipped.append((cell, f"no rerank variant exists for retriever={retriever}"))
            continue
        cells.append(cell)
    return cells, skipped


def run(overrides: list[str]) -> int:
    """Run every cell and write one run directory each."""
    base = load_config(overrides=overrides)
    matrix = typed_node(base, "ablation", AblationConfig)
    cells, skipped = expand(matrix)
    runs_dir = resolve_path(base, "paths.runs_dir")

    console.print(f"sweeping {len(cells)} configurations ({len(skipped)} excluded as meaningless)")
    written = 0
    for index, cell in enumerate(cells, start=1):
        cfg: DictConfig = load_config(overrides=[*overrides, *cell.overrides])
        prompts = {
            "generator": load_prompt(resolve_path(cfg, "paths.prompts_dir"), cfg.generator.prompt),
            "judge": load_prompt(resolve_path(cfg, "paths.prompts_dir"), cfg.judge.prompt),
        }
        client = None
        if cfg.generator.provider != "extractive" or cfg.judge.provider != "heuristic":
            client = build_model_client(cfg)

        try:
            stack = build_stack(cfg)
            outcome = evaluate(
                cfg,
                stack,
                build_generator(cfg, stack.tokenizer, client),
                build_judge(cfg, client),
                prompts,
            )
        except BackendUnavailableError as exc:
            # An optional dependency is missing (the reranker needs `make ml`).
            # Recording the cell as unmeasured and continuing beats dying at
            # cell 20 of 36; the report lists what was skipped and why.
            skipped.append((cell, f"backend unavailable: {exc}"))
            console.print(f"[yellow]{index}/{len(cells)} {cell}: skipped, {exc}[/yellow]")
            continue
        try:
            path = write_run(runs_dir, outcome.meta, outcome.rows, outcome.metrics, cfg)
        except RunExistsError:
            console.print(
                f"[yellow]{index}/{len(cells)} {cell}: already measured, skipping[/yellow]"
            )
            continue
        written += 1
        quality = outcome.metrics["composite_quality"]
        recall = outcome.metrics["recall_at_k"]
        p95_ms = outcome.metrics["p95_latency_s"] * 1000
        console.print(
            f"{index}/{len(cells)} {cell}: quality {quality:.3f} "
            f"recall@{cell.k} {recall:.3f} p95 {p95_ms:.1f}ms -> {path.name}"
        )

    for cell, reason in skipped:
        console.print(f"[dim]skipped {cell}: {reason}[/dim]")
    console.print(f"{written} runs written to {runs_dir}")
    return 0
