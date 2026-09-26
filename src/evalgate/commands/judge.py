"""``evalgate judge``: score answers and write the judgments to JSONL.

Scores every answer once, and -- when the position-swap probe is enabled --
a second time with the retrieved context in the opposite order. The two
scorings are stored side by side under different variants so the report can ask
whether the judge's opinion survived a change that cannot affect quality.

Each judge writes its own file under `evalsets.judgments_dir`, named after its
model, together with a record of the stack it judged over. Scoring with the
rule-based baseline therefore never replaces an LLM judge's scores, and the
calibration report describes each set by what produced it.
"""

from __future__ import annotations

from omegaconf import DictConfig
from rich.console import Console

from evalgate.config import JudgeConfig, ProbesConfig, load_config, resolve_path, typed_node
from evalgate.errors import EvalgateError
from evalgate.evalsets.judgments import JudgmentsMeta, load_judgment_set, write_judgments
from evalgate.evalsets.schemas import AnswerExample, JudgeScore
from evalgate.evalsets.store import read_jsonl
from evalgate.judging.base import JudgeInput
from evalgate.judging.factory import build_judge
from evalgate.models.client import build_model_client
from evalgate.pipeline import build_stack, describe_stack
from evalgate.runs.store import load_run_answers

console = Console()

VARIANT_PRIMARY = "primary"
VARIANT_SWAPPED = "swapped"
SOURCE_REFERENCE = "reference"
SOURCE_RUN = "run"


def _answers(
    cfg: DictConfig, settings: JudgeConfig, examples: list[AnswerExample]
) -> dict[str, str]:
    """The answer text to grade for each example."""
    if settings.answers_from == SOURCE_REFERENCE:
        return {example.id: example.reference_answer for example in examples}
    if settings.answers_from == SOURCE_RUN:
        rows = load_run_answers(resolve_path(cfg, "paths.runs_dir"), settings.run_id)
        return {str(row["example_id"]): str(row["answer"]) for row in rows}
    raise ValueError(f"judge.answers_from must be {SOURCE_REFERENCE!r} or {SOURCE_RUN!r}")


def run(overrides: list[str]) -> int:
    """Grade the answer eval set with the configured judge."""
    cfg = load_config(overrides=overrides)
    settings = typed_node(cfg, "judge", JudgeConfig)
    probes = typed_node(cfg, "probes", ProbesConfig)

    examples = read_jsonl(resolve_path(cfg, "evalsets.answers_path"), AnswerExample)
    if not examples:
        console.print("[red]no answer examples; run `make seed`[/red]")
        return 2

    answers = _answers(cfg, settings, examples)
    generator = (
        SOURCE_REFERENCE if settings.answers_from == SOURCE_REFERENCE else str(cfg.generator.model)
    )
    client = build_model_client(cfg) if settings.provider != "heuristic" else None
    judge = build_judge(cfg, client)
    stack = build_stack(cfg)

    console.print(
        f"judging {len(examples)} answers with {judge.model} "
        f"(provider {settings.provider}, api mode {cfg.api.mode})"
    )

    scores: list[JudgeScore] = []
    for example in examples:
        answer = answers.get(example.id)
        if answer is None:
            continue
        context = stack.retriever.retrieve(example.question)
        scores.append(
            judge.score(
                JudgeInput(
                    example_id=example.id,
                    question=example.question,
                    answer=answer,
                    context=context,
                    gold_spans=example.gold_spans,
                    variant=VARIANT_PRIMARY,
                    generator=generator,
                )
            )
        )
        if probes.position_swap:
            scores.append(
                judge.score(
                    JudgeInput(
                        example_id=example.id,
                        question=example.question,
                        answer=answer,
                        context=list(reversed(context)),
                        gold_spans=example.gold_spans,
                        variant=VARIANT_SWAPPED,
                        generator=generator,
                    )
                )
            )

    directory = resolve_path(cfg, "evalsets.judgments_dir")
    stack_description = describe_stack(cfg, stack)
    # The same judge's scores for other generators' answers are kept: the
    # self-preference probe compares one judge across two generators, and
    # rewriting the file on every run meant it could never see two.
    previous = load_judgment_set(directory, judge.model)
    kept = [item for item in previous.judgments if item.generator != generator] if previous else []
    previous_stack = previous.meta.stack if previous and previous.meta else None
    if kept and previous_stack is not None and previous_stack != stack_description:
        raise EvalgateError(
            f"{judge.model} has scores for other generators judged over a different "
            f"stack ({previous_stack}); scores over two stacks cannot share a file "
            "or be compared by the self-preference probe. Re-judge them under this "
            "stack, or move the old file aside."
        )
    generators = sorted({item.generator for item in kept} | {generator})

    meta = JudgmentsMeta(
        judge_model=judge.model,
        judge_provider=settings.provider,
        judge_prompt=settings.prompt,
        # From the scores themselves: the rule-based judge reads no prompt, and
        # its records say so with an empty hash.
        judge_prompt_hash=scores[0].prompt_hash if scores else "",
        axes=list(settings.axes),
        scale_min=settings.scale_min,
        scale_max=settings.scale_max,
        answers_from=settings.answers_from,
        generators=generators,
        api_mode=str(cfg.api.mode),
        stack=stack_description,
    )
    path = write_judgments(directory, meta, [*kept, *scores])
    primary = [score for score in scores if score.variant == VARIANT_PRIMARY]
    tokens = sum(score.input_tokens + score.output_tokens for score in scores)
    console.print(
        f"{len(primary)} primary judgments "
        f"({len(scores) - len(primary)} swapped-context repeats), "
        f"{tokens} tokens -> {path}"
    )
    return 0
