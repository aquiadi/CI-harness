"""Pareto figures.

Two series only: the non-dominated configurations, which are the subject, and
the dominated ones, which are context. Context is drawn in the muted ink used
for axis labels rather than a second categorical hue -- it is not a second
identity, it is the background the frontier stands out from. The palette
validator flags that gray for low chroma, which is exactly what a recessive
mark is supposed to be; separation from the accent passes in both modes (CVD
delta E 15.9, normal-vision 17.8 light / 17.0 dark) and both clear 3:1 against
their surface.

Identity is never colour alone: frontier points are larger, joined by a step
line, and directly labelled, and both series appear in the legend.

Each figure is rendered twice, for a light and a dark surface, and the report
selects between them with a `<picture>` element, so the figure does not glare
on a dark page.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

DARK_SUFFIX = "_dark"


@dataclass(frozen=True, slots=True)
class Surface:
    """The colours one rendering of a figure is drawn against."""

    name: str
    background: str
    accent: str
    context: str
    ink: str
    muted: str
    grid: str
    axis: str


LIGHT = Surface(
    name="light",
    background="#fcfcfb",
    accent="#2a78d6",
    context="#898781",
    ink="#0b0b0b",
    muted="#52514e",
    grid="#e1e0d9",
    axis="#c3c2b7",
)
DARK = Surface(
    name="dark",
    background="#1a1a19",
    accent="#3987e5",
    context="#898781",
    ink="#ffffff",
    muted="#c3c2b7",
    grid="#2c2c2a",
    axis="#383835",
)
SURFACES = (LIGHT, DARK)


@dataclass(frozen=True, slots=True)
class Point:
    """One configuration on the frontier plot."""

    label: str
    x: float
    y: float
    dominated: bool


def frontier(points: Sequence[Point]) -> list[Point]:
    """The non-dominated set: maximise y, minimise x.

    A point is dominated when another is at least as good on both axes and
    strictly better on one. Ties on both axes are kept, so two configurations
    that measure identically both appear rather than one vanishing by sort
    order.
    """
    keep: list[Point] = []
    for candidate in points:
        beaten = any(
            other.x <= candidate.x
            and other.y >= candidate.y
            and (other.x < candidate.x or other.y > candidate.y)
            for other in points
        )
        if not beaten:
            keep.append(candidate)
    return sorted(keep, key=lambda point: (point.x, -point.y))


def render(
    points: Sequence[Point],
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    x_log: bool = False,
) -> list[Path]:
    """Write the light and dark renderings of one frontier figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    best = frontier(points)
    best_labels = {point.label for point in best}
    written: list[Path] = []

    for surface in SURFACES:
        figure, axes = plt.subplots(figsize=(7.2, 4.5), dpi=160)
        figure.patch.set_facecolor(surface.background)
        axes.set_facecolor(surface.background)

        dominated = [point for point in points if point.label not in best_labels]
        axes.scatter(
            [point.x for point in dominated],
            [point.y for point in dominated],
            s=34,
            c=surface.context,
            edgecolors=surface.background,
            linewidths=0.8,
            label=f"dominated ({len(dominated)})",
            zorder=2,
        )
        axes.step(
            [point.x for point in best],
            [point.y for point in best],
            where="post",
            color=surface.accent,
            linewidth=2.0,
            alpha=0.55,
            zorder=3,
        )
        axes.scatter(
            [point.x for point in best],
            [point.y for point in best],
            s=76,
            c=surface.accent,
            edgecolors=surface.background,
            linewidths=1.4,
            label=f"non-dominated ({len(best)})",
            zorder=4,
        )
        # Frontier points cluster where the frontier is steep, so labels are
        # staggered above and below alternately; a fixed offset collides.
        for index, point in enumerate(best):
            axes.annotate(
                point.label,
                (point.x, point.y),
                textcoords="offset points",
                xytext=(8, 6) if index % 2 == 0 else (8, -12),
                fontsize=7.5,
                color=surface.muted,
                zorder=5,
            )

        # A log axis needs strictly positive values; fall back rather than
        # emitting a warning and an empty axis.
        if x_log and all(point.x > 0 for point in points):
            axes.set_xscale("log")
        axes.set_title(title, color=surface.ink, fontsize=11, loc="left", pad=12)
        axes.set_xlabel(x_label, color=surface.muted, fontsize=9)
        axes.set_ylabel(y_label, color=surface.muted, fontsize=9)
        axes.tick_params(colors=surface.muted, labelsize=8, length=0)
        axes.grid(True, color=surface.grid, linewidth=0.6, alpha=0.9)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(surface.axis)
            axes.spines[side].set_linewidth(0.8)

        # Frontier labels extend to the right of their point; without extra
        # margin the rightmost one runs off the figure.
        axes.margins(x=0.12, y=0.10)
        legend = axes.legend(frameon=False, fontsize=8, loc="lower right")
        for text in legend.get_texts():
            text.set_color(surface.muted)

        figure.tight_layout()
        target = path if surface is LIGHT else path.with_stem(path.stem + DARK_SUFFIX)
        figure.savefig(target, facecolor=surface.background)
        plt.close(figure)
        written.append(target)
    return written
