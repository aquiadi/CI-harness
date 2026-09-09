"""The Makefile is the interface, so its wiring is worth testing.

A flag a user passes and the Makefile silently drops is the same class of
defect as a metric the gate silently skips: the command appears to work and
does something else. These tests are cheap and they caught a real one --
`make gen-eval PROFILE=...` ignored PROFILE and ran the default stack.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# A recipe line that runs the pipeline. `serve` and `docker` are excluded by
# construction: they take configuration through the environment and the image,
# not through hydra overrides.
INVOCATION = re.compile(r"^\t.*(?:evalgate\.cli\s+[\w-]+|scripts/\S+\.py)")
# A target, not a variable assignment: `RUN := $(UV) run` must not look like one.
TARGET = re.compile(r"^(?P<name>[a-zA-Z][\w-]*)\s*:(?!=)")
# `record` pins its own experiment on purpose: it exists to record the cassettes
# for the live configuration, so taking PROFILE would defeat the point.
PINS_ITS_OWN_PROFILE = frozenset({"record"})


def recipe_lines(makefile: str) -> list[tuple[str, str]]:
    """Every recipe line that runs the CLI or a script, with its target."""
    lines: list[tuple[str, str]] = []
    target = ""
    for line in makefile.splitlines():
        match = TARGET.match(line)
        if match:
            target = match.group("name")
        elif INVOCATION.match(line):
            lines.append((target, line.strip()))
    return lines


@pytest.fixture(scope="module")
def makefile(repo_root: Path) -> str:
    return (repo_root / "Makefile").read_text(encoding="utf-8")


def test_every_pipeline_target_passes_profile(makefile: str) -> None:
    """A dropped PROFILE runs the wrong corpus, embedder and models in silence."""
    offenders = [
        target
        for target, line in recipe_lines(makefile)
        if target not in PINS_ITS_OWN_PROFILE and "$(PROFILE)" not in line
    ]
    assert not offenders, f"these targets ignore PROFILE: {offenders}"


def test_every_pipeline_target_passes_args(makefile: str) -> None:
    """ARGS is how a user overrides anything; dropping it is equally silent."""
    offenders = [target for target, line in recipe_lines(makefile) if "$(ARGS)" not in line]
    assert not offenders, f"these targets ignore ARGS: {offenders}"


def test_profile_defaults_to_the_frozen_baseline(makefile: str) -> None:
    """`make eval` with no arguments must measure what baseline.json froze."""
    assert "PROFILE ?= +experiment=baseline" in makefile


def test_every_target_is_declared_phony(makefile: str) -> None:
    """A target sharing a name with a directory silently stops running."""
    # .PHONY spans several lines with backslash continuations; join them first
    # or half the declarations are invisible.
    joined = re.sub(r"\\\n\s*", " ", makefile)
    declared = set()
    for line in joined.splitlines():
        if line.startswith(".PHONY:"):
            declared.update(line.removeprefix(".PHONY:").split())
    targets = {
        match.group("name") for line in makefile.splitlines() if (match := TARGET.match(line))
    }
    assert targets <= declared, f"not declared .PHONY: {sorted(targets - declared)}"
