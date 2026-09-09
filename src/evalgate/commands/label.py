"""``evalgate label``: the terminal labelling CLI.

Two modes. ``label.mode=answers`` shows a question, the context that was
retrieved for it and the answer under review, and takes a 1-5 rating on each of
groundedness, relevance and citation correctness. ``label.mode=retrieval``
reviews generated retrieval candidates and marks them accepted or rejected.

Three properties matter more than looks:

*Nothing is lost.* Every rating is appended and fsynced the moment it is given.
Ctrl-C, a closed lid or a dropped connection costs the item in progress and
nothing else.

*It resumes.* On start it reads what is already labelled and skips it, so a
150-item set can be done in six sittings.

*It records who and when.* Judge validation is a comparison against a specific
person's labels at a specific time; an unattributed label cannot be argued
with later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from evalgate.config import JudgeConfig, LabelConfig, load_config, resolve_path, typed_node
from evalgate.corpus.manifest import utc_now_iso
from evalgate.evalsets.schemas import (
    AnswerExample,
    AxisScores,
    HumanLabel,
    RetrievalExample,
    ReviewStatus,
)
from evalgate.evalsets.store import append_jsonl, read_jsonl, write_jsonl
from evalgate.hashing import sha256_text, short
from evalgate.pipeline import RetrievalStack, build_stack

console = Console()

QUIT = "q"
SKIP = "s"
NOTE = "n"
ACCEPT = "a"
REJECT = "r"

MODE_ANSWERS = "answers"
MODE_RETRIEVAL = "retrieval"
SOURCE_REFERENCE = "reference"
SOURCE_RUN = "run"


class _QuitError(Exception):
    """Raised internally when the labeller asks to stop."""


@dataclass(frozen=True, slots=True)
class Item:
    """One thing to be rated: a question and the answer text under review."""

    example: AnswerExample
    answer: str
    answer_hash: str


def _ask(prompt: str, choices: list[str]) -> str:
    """Ask for one keystroke-sized answer, treating interruption as quit."""
    try:
        return Prompt.ask(prompt, choices=choices, show_choices=True)
    except (KeyboardInterrupt, EOFError) as exc:
        raise _QuitError from exc


def _ask_score(axis: str, low: int, high: int) -> int | None:
    """Ask for one axis score; None means skip this item."""
    choices = [str(value) for value in range(low, high + 1)] + [SKIP, QUIT]
    answer = _ask(f"  {axis} ({low}-{high})", choices)
    if answer == QUIT:
        raise _QuitError
    if answer == SKIP:
        return None
    return int(answer)


def _render_context(stack: RetrievalStack, question: str, k: int) -> None:
    """Show the retrieved context the answer was, or would have been, given."""
    table = Table(title=f"retrieved context (k={k})", show_edge=False, show_lines=True)
    table.add_column("chunk", no_wrap=True)
    table.add_column("score", justify="right")
    table.add_column("text")
    for result in stack.retriever.retrieve(question, k=k):
        # Text() rather than a string: chunk ids and citations are written in
        # square brackets, which rich would otherwise read as markup and
        # silently delete -- taking the citations the labeller is rating with
        # them.
        table.add_row(result.chunk_id, f"{result.score:.4f}", Text(" ".join(result.text.split())))
    console.print(table)


def _label_answers(cfg: DictConfig, label: LabelConfig) -> int:
    """Rate answers on the three axes."""
    answers_path = resolve_path(cfg, "evalsets.answers_path")
    labels_path = resolve_path(cfg, "evalsets.human_labels_path")
    examples = read_jsonl(answers_path, AnswerExample)
    if not examples:
        console.print(f"[red]no answer examples in {answers_path}[/red]")
        return 2

    existing = read_jsonl(labels_path, HumanLabel)
    done = {(record.example_id, record.answer_hash) for record in existing}

    items = _build_items(cfg, label, examples)
    pending = [
        item for item in items if label.relabel or (item.example.id, item.answer_hash) not in done
    ]
    if not pending:
        console.print(f"[green]all {len(items)} items already labelled[/green] ({labels_path})")
        return 0

    judge = typed_node(cfg, "judge", JudgeConfig)
    console.print(
        f"labelling {len(pending)} of {len(items)} items as [bold]{label.labeller}[/bold]; "
        f"{len(done)} already done. Ctrl-C or 'q' stops and keeps everything rated so far."
    )
    stack = build_stack(cfg)

    written = 0
    try:
        for position, item in enumerate(pending, start=1):
            written += _label_one(stack, label, judge, labels_path, item, position, len(pending))
    except _QuitError:
        console.print("\n[yellow]stopped[/yellow]")
    console.print(f"[green]{written} labels written[/green] to {labels_path}")
    return 0


def _label_one(
    stack: RetrievalStack,
    label: LabelConfig,
    judge: JudgeConfig,
    labels_path: Path,
    item: Item,
    position: int,
    total: int,
) -> int:
    """Rate one item, appending the label immediately."""
    console.rule(f"{position}/{total}  {item.example.id}")
    console.print(Panel(Text(item.example.question), title="question"))
    _render_context(stack, item.example.question, label.context_k)
    if label.show_gold and item.example.gold_spans:
        console.print(Panel(Text("\n\n".join(item.example.gold_spans)), title="gold evidence"))
    console.print(
        Panel(Text(item.answer), title=f"answer under review ({short(item.answer_hash)})")
    )

    started = time.monotonic()
    scores: dict[str, int] = {}
    for axis in judge.axes:
        value = _ask_score(axis, judge.scale_min, judge.scale_max)
        if value is None:
            console.print("[yellow]skipped[/yellow]")
            return 0
        scores[axis] = value

    note = None
    if _ask("  add a note?", ["y", "n"]) == "y":
        note = Prompt.ask("  note")

    append_jsonl(
        labels_path,
        HumanLabel(
            example_id=item.example.id,
            answer_hash=item.answer_hash,
            scores=AxisScores(**scores),
            notes=note,
            labeller=label.labeller,
            labelled_at=utc_now_iso(),
            seconds_spent=round(time.monotonic() - started, 1),
        ),
    )
    return 1


def _build_items(cfg: DictConfig, label: LabelConfig, examples: list[AnswerExample]) -> list[Item]:
    """Pair each example with the answer text to be rated."""
    if label.answers_from == SOURCE_REFERENCE:
        return [
            Item(
                example=example,
                answer=example.reference_answer,
                answer_hash=sha256_text(example.reference_answer),
            )
            for example in examples
        ]
    if label.answers_from == SOURCE_RUN:
        from evalgate.runs.store import load_run_answers

        answers = load_run_answers(resolve_path(cfg, "paths.runs_dir"), label.run_id)
        by_id = {record["example_id"]: str(record["answer"]) for record in answers}
        return [
            Item(
                example=example,
                answer=by_id[example.id],
                answer_hash=sha256_text(by_id[example.id]),
            )
            for example in examples
            if example.id in by_id
        ]
    raise ValueError(f"label.answers_from must be {SOURCE_REFERENCE!r} or {SOURCE_RUN!r}")


def _review_retrieval(cfg: DictConfig, label: LabelConfig) -> int:
    """Accept or reject generated retrieval candidates."""
    path = resolve_path(cfg, "evalsets.retrieval_path")
    examples = read_jsonl(path, RetrievalExample)
    if not examples:
        console.print(f"[red]no retrieval candidates in {path}[/red]")
        return 2

    pending = [
        example for example in examples if label.relabel or example.status == ReviewStatus.DRAFT
    ]
    if not pending:
        counts = {status: sum(1 for e in examples if e.status == status) for status in ReviewStatus}
        console.print(f"[green]nothing left to review[/green]: {dict(counts)}")
        return 0

    console.print(
        f"reviewing {len(pending)} of {len(examples)} candidates as [bold]{label.labeller}[/bold]. "
        "'a' accept, 'r' reject, 's' skip, 'q' stop."
    )
    reviewed = 0
    try:
        for position, example in enumerate(pending, start=1):
            console.rule(f"{position}/{len(pending)}  {example.id}  [{example.difficulty}]")
            console.print(Panel(Text(example.question), title="question"))
            console.print(
                Panel(
                    Text("\n\n".join(example.gold_spans)),
                    title=f"gold evidence in {example.gold_doc_id}",
                )
            )
            answer = _ask("  verdict", [ACCEPT, REJECT, SKIP, QUIT])
            if answer == QUIT:
                raise _QuitError
            if answer == SKIP:
                continue
            example.status = ReviewStatus.ACCEPTED if answer == ACCEPT else ReviewStatus.REJECTED
            example.reviewed_by = label.labeller
            example.reviewed_at = utc_now_iso()
            # Rewritten after every decision: the file is small, and an
            # interrupted session must leave the accepted ones accepted.
            write_jsonl(path, examples)
            reviewed += 1
    except _QuitError:
        console.print("\n[yellow]stopped[/yellow]")

    accepted = sum(1 for example in examples if example.status == ReviewStatus.ACCEPTED)
    console.print(f"[green]{reviewed} reviewed[/green]; {accepted} accepted in total ({path})")
    return 0


def run(overrides: list[str]) -> int:
    """Dispatch to the answer or retrieval labelling loop."""
    cfg = load_config(overrides=overrides)
    label = typed_node(cfg, "label", LabelConfig)
    if label.mode == MODE_RETRIEVAL:
        return _review_retrieval(cfg, label)
    if label.mode == MODE_ANSWERS:
        return _label_answers(cfg, label)
    console.print(f"[red]label.mode must be {MODE_ANSWERS!r} or {MODE_RETRIEVAL!r}[/red]")
    return 2
