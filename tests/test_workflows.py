"""The CI workflows invoke the Makefile; their wiring is worth testing too.

The nightly live job ran `make eval ARGS="+experiment=live"` for as long as it
existed. The Makefile already passes `+experiment=baseline` as PROFILE, so the
command line carried two `+experiment` values and hydra refused to compose it
-- every night, before a single measurement. Nothing noticed, because the only
place that command ran was the job itself. These tests run what the workflows
would run, as far as that is possible without a network or a key.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalgate.config import load_config

WORKFLOWS = ("ci.yml", "nightly.yml")
MAKE = re.compile(r"(?:^|[;&|]\s*|\n\s*)make\s+(?P<target>[a-z][\w-]*)")
# `${{ vars.NAME || '+experiment=x' }}` -- the fallback is what runs when the
# repository variable is unset.
FALLBACK = re.compile(r"\|\|\s*'(?P<value>[^']*)'")
EXPERIMENT = re.compile(r"\+experiment=(?P<name>\w+)")


def _load(repo_root: Path, name: str) -> dict[str, Any]:
    payload: dict[str, Any] = yaml.safe_load(
        (repo_root / ".github" / "workflows" / name).read_text(encoding="utf-8")
    )
    return payload


def _steps(workflow: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every (job, step) pair in a workflow."""
    return [
        (job, step) for job in workflow.get("jobs", {}).values() for step in job.get("steps", [])
    ]


def _profiles(workflow: dict[str, Any]) -> list[str]:
    """Every PROFILE value a workflow can run with, fallback expressions resolved."""
    values: list[str] = []
    envs = [job.get("env", {}) for job in workflow.get("jobs", {}).values()]
    envs += [step.get("env", {}) for _, step in _steps(workflow)]
    for env in envs:
        raw = env.get("PROFILE")
        if raw is None:
            continue
        match = FALLBACK.search(str(raw))
        values.append(match.group("value") if match else str(raw))
    return values


@pytest.mark.parametrize("name", WORKFLOWS)
def test_no_workflow_passes_an_experiment_through_args(repo_root: Path, name: str) -> None:
    """ARGS is appended after PROFILE; an experiment there is a second one."""
    for _, step in _steps(_load(repo_root, name)):
        run = str(step.get("run", ""))
        assert not re.search(r"ARGS=[\"']?[^\"'\n]*\+experiment", run), (
            f"{name}: {step.get('name')!r} passes +experiment through ARGS; use PROFILE"
        )


@pytest.mark.parametrize("name", WORKFLOWS)
def test_every_workflow_profile_composes(repo_root: Path, name: str) -> None:
    for profile in _profiles(_load(repo_root, name)):
        load_config(overrides=profile.split())


@pytest.mark.parametrize("name", WORKFLOWS)
def test_every_make_target_a_workflow_runs_exists(repo_root: Path, name: str) -> None:
    makefile = (repo_root / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-zA-Z][\w-]*)\s*:(?!=)", makefile, flags=re.MULTILINE))
    for _, step in _steps(_load(repo_root, name)):
        for match in MAKE.finditer(str(step.get("run", ""))):
            assert match.group("target") in targets, f"{name}: no make target {match[0]!r}"


def test_the_nightly_measures_a_live_profile(repo_root: Path) -> None:
    """The nightly is the half of the split that talks to a model."""
    profiles = _profiles(_load(repo_root, "nightly.yml"))
    assert profiles, "the nightly must set PROFILE explicitly"
    for profile in profiles:
        cfg = load_config(overrides=profile.split())
        assert cfg.api.mode == "live"
        assert cfg.corpus.name == "cbam", "the nightly exists to measure the real documents"


EXPERIMENTS = sorted(
    path.stem for path in (Path(__file__).parent.parent / "configs" / "experiment").glob("*.yaml")
)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_composes_after_the_makefile_default(experiment: str) -> None:
    """`make X PROFILE=+experiment=Y` replaces the default; it never stacks on it."""
    cfg = load_config(overrides=[f"+experiment={experiment}"])
    assert str(cfg.gate.baseline_path).endswith(".json")


def test_every_experiment_has_its_own_baseline(repo_root: Path) -> None:
    """Two profiles sharing a baseline file would gate one system against another."""
    paths: dict[str, str] = {}
    for path in sorted((repo_root / "configs" / "experiment").glob("*.yaml")):
        cfg = load_config(overrides=[f"+experiment={path.stem}"])
        paths[path.stem] = str(cfg.gate.baseline_path)
    assert len(set(paths.values())) == len(paths), paths
