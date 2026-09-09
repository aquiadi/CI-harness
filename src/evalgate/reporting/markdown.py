"""Small markdown helpers.

Reports carry no timestamp. A report is a pure function of the artifacts it
reads, so regenerating one whose inputs have not changed produces a
byte-identical file and leaves the working tree clean. Provenance is recorded
as the input hashes instead, which is what actually identifies the data.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

NA = "-"


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    """Render a markdown table."""
    body = list(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in body)
    return "\n".join(lines)


def _cell(value: object) -> str:
    if value is None:
        return NA
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", r"\|").replace("\n", " ")


def number(value: float | None, digits: int = 3, fallback: str = NA) -> str:
    """Format an optional number."""
    return fallback if value is None else f"{value:.{digits}f}"


def heading(text: str, level: int = 2) -> str:
    """A markdown heading."""
    return f"{'#' * level} {text}"


def bullets(items: Iterable[str]) -> str:
    """A markdown bullet list."""
    return "\n".join(f"- {item}" for item in items)


def truncate(text: str, limit: int) -> str:
    """Collapse whitespace and cut to a length, marking the cut."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
