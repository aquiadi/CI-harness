"""The judge calibration report.

The centrepiece of the repo, and the section a sceptical reader should turn to
first. It answers: does this judge agree with a person, how badly does it fail
when it fails, and does its score move for reasons that have nothing to do with
answer quality.

Every figure is computed from artifacts on disk. Where a figure cannot be
computed -- no labels, no variance, no second generator -- the report says so
and why, rather than printing a zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from evalgate.config import JudgeConfig, ProbesConfig, resolve_path, typed_node
from evalgate.evalsets.schemas import AnswerExample, HumanLabel, JudgeScore
from evalgate.evalsets.store import read_jsonl
from evalgate.evaluation.agreement import (
    AxisAgreement,
    Disagreement,
    axis_agreement,
    worst_disagreements,
)
from evalgate.evaluation.bias import length_bias, position_consistency, self_preference
from evalgate.hashing import hash_obj, short
from evalgate.ingest.tokens import TokenEstimator
from evalgate.reporting.markdown import bullets, heading, number, table, truncate

TITLE = "Judge calibration"
SEED_LABELLER = "seed-author"
ANSWER_EXCERPT_CHARS = 160
RATIONALE_EXCERPT_CHARS = 200
VARIANT_PRIMARY = "primary"
VARIANT_SWAPPED = "swapped"

INDEPENDENCE_WARNING = (
    "**These are not independent human labels.** The only labels available are "
    "`seed-author` ratings, authored alongside the answers they rate (see "
    "docs/DECISIONS.md D-0019). Agreement measured against them says almost nothing "
    "about whether the judge agrees with a person: the same author decided both what "
    "the answer would get wrong and what score that deserved. They exist so this "
    "pipeline is runnable and testable before anyone has labelled. Run `make label` "
    "and regenerate to get a number that means something."
)

MIXED_LABELS_NOTE = (
    "Labels come from more than one source. Independent human labels and `seed-author` "
    "ratings are reported separately; they are never pooled."
)


@dataclass(frozen=True, slots=True)
class LabelSet:
    """Labels from one source, and what that source is worth."""

    name: str
    independent: bool
    labels: list[HumanLabel]


def partition_labels(labels: Sequence[HumanLabel]) -> list[LabelSet]:
    """Split labels by labeller, keeping the non-independent ones separate."""
    seeded = [label for label in labels if label.labeller == SEED_LABELLER]
    human = [label for label in labels if label.labeller != SEED_LABELLER]
    sets: list[LabelSet] = []
    if human:
        labellers = sorted({label.labeller for label in human})
        sets.append(LabelSet(name=", ".join(labellers), independent=True, labels=human))
    if seeded:
        sets.append(LabelSet(name=SEED_LABELLER, independent=False, labels=seeded))
    return sets


def _pair_scores(
    label_set: LabelSet, judgments: Sequence[JudgeScore], axis: str
) -> tuple[list[int], list[int], list[Disagreement]]:
    """Align human and judge scores on the same (example, answer) pairs.

    Joining on the answer hash as well as the example id is the point: a judge
    score for one answer and a human label for a different answer to the same
    question are not comparable, and silently pairing them would manufacture
    agreement.
    """
    by_key = {
        (judgment.example_id, judgment.answer_hash): judgment
        for judgment in judgments
        if judgment.variant == VARIANT_PRIMARY
    }
    human_scores: list[int] = []
    judge_scores: list[int] = []
    disagreements: list[Disagreement] = []
    for label in sorted(label_set.labels, key=lambda item: item.example_id):
        judgment = by_key.get((label.example_id, label.answer_hash))
        if judgment is None:
            continue
        human_value = label.scores.as_dict()[axis]
        judge_value = judgment.scores.as_dict()[axis]
        human_scores.append(human_value)
        judge_scores.append(judge_value)
        if human_value != judge_value:
            disagreements.append(
                Disagreement(
                    example_id=label.example_id,
                    axis=axis,
                    human=human_value,
                    judge=judge_value,
                    gap=abs(human_value - judge_value),
                    rationale=judgment.rationales.get(axis, ""),
                )
            )
    return human_scores, judge_scores, disagreements


def _agreement_table(rows: Sequence[AxisAgreement]) -> str:
    return table(
        [
            "axis",
            "n",
            "kappa",
            "quadratic kappa",
            "exact",
            "within 1",
            "mean human",
            "mean judge",
            "mean signed error",
        ],
        [
            [
                row.axis,
                row.n,
                number(row.kappa) if row.is_defined else f"undefined ({row.undefined_reason})",
                number(row.quadratic_kappa) if row.is_defined else "undefined",
                row.exact_agreement,
                row.within_one,
                row.mean_human,
                row.mean_judge,
                row.mean_signed_error,
            ]
            for row in rows
        ],
    )


def _confusion_section(row: AxisAgreement, scale_min: int, scale_max: int) -> str:
    labels = list(range(scale_min, scale_max + 1))
    headers = ["human \\ judge", *[str(value) for value in labels]]
    rows = [[str(labels[index]), *counts] for index, counts in enumerate(row.confusion)]
    return f"{heading(f'Confusion matrix: {row.axis}', 4)}\n\n{table(headers, rows)}"


def _probe_section(
    cfg: DictConfig,
    judgments: Sequence[JudgeScore],
    answers: Sequence[AnswerExample],
    tokenizer: TokenEstimator,
    axes: Sequence[str],
    probes: ProbesConfig,
) -> str:
    parts: list[str] = [heading("Bias probes", 3)]

    primary = [item for item in judgments if item.variant == VARIANT_PRIMARY]
    swapped = {item.example_id: item for item in judgments if item.variant == VARIANT_SWAPPED}

    parts.append(heading("Position swap: does reversing the context change the score?", 4))
    if not probes.position_swap or not swapped:
        parts.append("Not run: no swapped-context judgments in the judge score file.")
    else:
        rows = []
        for axis in axes:
            paired = [
                (item, swapped[item.example_id]) for item in primary if item.example_id in swapped
            ]
            consistency = position_consistency(
                axis,
                [item.scores.as_dict()[axis] for item, _ in paired],
                [other.scores.as_dict()[axis] for _, other in paired],
            )
            rows.append(
                [
                    axis,
                    consistency.n,
                    consistency.exact_agreement,
                    consistency.mean_absolute_difference,
                    consistency.max_absolute_difference,
                ]
            )
        parts.append(
            table(
                ["axis", "n", "identical score", "mean abs difference", "max abs difference"], rows
            )
        )
        parts.append(
            "The answer is unchanged between the two scorings; only the order of the "
            "retrieved context differs. Any movement is the judge responding to "
            "presentation rather than to quality."
        )

    parts.append(heading("Length bias: do longer answers score better?", 4))
    if not probes.length_bias:
        parts.append("Disabled in config.")
    else:
        lengths_by_id = {
            example.id: tokenizer.count(example.reference_answer) for example in answers
        }
        rows = []
        for axis in axes:
            usable = [item for item in primary if item.example_id in lengths_by_id]
            bias = length_bias(
                axis,
                [lengths_by_id[item.example_id] for item in usable],
                [item.scores.as_dict()[axis] for item in usable],
            )
            rows.append(
                [
                    axis,
                    bias.n,
                    number(bias.spearman),
                    number(bias.p_value),
                    bias.mean_tokens,
                    bias.note or "",
                ]
            )
        parts.append(table(["axis", "n", "spearman rho", "p", "mean answer tokens", "note"], rows))
        parts.append(
            "A strong positive correlation is the failure mode that rewards padding. "
            "It is evidence, not proof: longer answers may genuinely be better."
        )

    parts.append(heading("Self-preference: does the judge favour its own model family?", 4))
    parts.append(_self_preference_section(cfg, probes, judgments, axes))
    return "\n\n".join(parts)


def _self_preference_section(
    cfg: DictConfig, probes: ProbesConfig, judgments: Sequence[JudgeScore], axes: Sequence[str]
) -> str:
    """Compare one judge's scores across answers from two generators."""
    if not probes.self_preference:
        return "Disabled in config."
    by_generator: dict[str, list[JudgeScore]] = {}
    for judgment in judgments:
        if judgment.variant != VARIANT_PRIMARY:
            continue
        by_generator.setdefault(judgment.generator, []).append(judgment)
    if len(by_generator) < 2:
        return (
            "Not run: this probe needs answers to the same questions from two "
            f"generators (config `probes.contrast_generator={probes.contrast_generator}`), "
            "graded by the same judge. Produce them with `make ablate` and regenerate."
        )
    judge_model = judgments[0].judge_model if judgments else "unknown"
    rows = []
    for axis in axes:
        result = self_preference(
            axis,
            judge_model,
            {
                name: [item.scores.as_dict()[axis] for item in items]
                for name, items in by_generator.items()
            },
        )
        rows.append(
            [
                axis,
                result.n,
                *[number(value) for value in result.means.values()],
                number(result.gap),
                result.favoured or "-",
            ]
        )
    headers = ["axis", "n", *sorted(by_generator), "gap", "favoured"]
    return table(headers, rows)


def _prompt_hash_label(judgments: Sequence[JudgeScore]) -> str:
    """Short prompt hash of the judgments, or a marker when there is none."""
    if judgments and judgments[0].prompt_hash:
        return short(judgments[0].prompt_hash)
    return "n/a"


def _inputs_digest(judgments: Sequence[JudgeScore], labels: Sequence[HumanLabel]) -> str:
    """Identity of the data this report was computed from.

    Stands in for a timestamp: a report is a pure function of its inputs, so
    what identifies it is what went in, not when it was rendered.
    """
    payload = [
        [item.model_dump(mode="json") for item in judgments],
        [item.model_dump(mode="json") for item in labels],
    ]
    return short(hash_obj(payload))


def build_report(
    cfg: DictConfig,
    tokenizer: TokenEstimator,
    stack_summary: dict[str, str],
) -> str:
    """Render the calibration report from the artifacts on disk."""
    settings = typed_node(cfg, "judge", JudgeConfig)
    probes = typed_node(cfg, "probes", ProbesConfig)
    axes = list(settings.axes)

    judgments = read_jsonl(resolve_path(cfg, "evalsets.judge_scores_path"), JudgeScore)
    answers = read_jsonl(resolve_path(cfg, "evalsets.answers_path"), AnswerExample)
    labels = read_jsonl(resolve_path(cfg, "evalsets.human_labels_path"), HumanLabel)
    seed_labels = read_jsonl(resolve_path(cfg, "evalsets.seed_labels_path"), HumanLabel)
    label_sets = partition_labels([*labels, *seed_labels])

    n_primary = len([item for item in judgments if item.variant == VARIANT_PRIMARY])
    n_swapped = len([item for item in judgments if item.variant == VARIANT_SWAPPED])
    provenance = [
        f"judge: `{settings.model}` (provider `{settings.provider}`)",
        f"judge prompt: `{settings.prompt}` ({_prompt_hash_label(judgments)})",
        f"scale: {settings.scale_min}-{settings.scale_max}",
        f"axes: {', '.join(axes)}",
        *[f"{key}: {value}" for key, value in sorted(stack_summary.items())],
        f"judgments: {n_primary} primary, {n_swapped} swapped-context",
        f"answers graded from: `{settings.answers_from}`",
        f"inputs digest: {_inputs_digest(judgments, [*labels, *seed_labels])}",
    ]
    parts: list[str] = [heading(TITLE, 1), bullets(provenance)]

    if not judgments:
        parts.append(
            "\nNo judgments on disk. Run `make judge` (add `judge=heuristic` to run the "
            "rule-based baseline with no credentials) and regenerate."
        )
        return "\n\n".join(parts) + "\n"

    if not label_sets:
        parts.append("\nNo labels on disk. Run `make label`, then regenerate this report.")
        return "\n\n".join(parts) + "\n"

    if len(label_sets) > 1:
        parts.append(MIXED_LABELS_NOTE)

    for label_set in label_sets:
        parts.append(heading(f"Agreement with {label_set.name} labels", 2))
        if not label_set.independent:
            parts.append(INDEPENDENCE_WARNING)

        rows: list[AxisAgreement] = []
        every_disagreement: list[Disagreement] = []
        for axis in axes:
            human_scores, judge_scores, disagreements = _pair_scores(label_set, judgments, axis)
            rows.append(
                axis_agreement(
                    axis, human_scores, judge_scores, settings.scale_min, settings.scale_max
                )
            )
            every_disagreement.extend(disagreements)

        parts.append(_agreement_table(rows))
        parts.append(
            "`mean signed error` is judge minus human: positive means the judge is more "
            "generous than the labeller."
        )
        for row in rows:
            parts.append(_confusion_section(row, settings.scale_min, settings.scale_max))

        parts.append(heading(f"Worst disagreements ({label_set.name})", 3))
        worst = worst_disagreements(every_disagreement, probes.worst_disagreements)
        if not worst:
            parts.append("None: the judge matched every label exactly.")
        else:
            answers_by_id = {example.id: example for example in answers}
            parts.append(
                table(
                    ["example", "axis", "human", "judge", "gap", "judge rationale", "answer"],
                    [
                        [
                            item.example_id,
                            item.axis,
                            item.human,
                            item.judge,
                            item.gap,
                            truncate(item.rationale, RATIONALE_EXCERPT_CHARS),
                            truncate(
                                answers_by_id[item.example_id].reference_answer
                                if item.example_id in answers_by_id
                                else "",
                                ANSWER_EXCERPT_CHARS,
                            ),
                        ]
                        for item in worst
                    ],
                )
            )

    parts.append(_probe_section(cfg, judgments, answers, tokenizer, axes, probes))
    return "\n\n".join(parts) + "\n"


def write_report(path: Path, content: str) -> bool:
    """Write the report, reporting whether anything changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True
