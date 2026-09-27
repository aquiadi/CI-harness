"""The committed reports are what the committed artifacts render to.

The README has had this guarantee since D-0035. The reports did not, and the
pareto report sat at 63 configurations while `runs/` held 65 and the README --
rendered from the same runs -- said 65. A report that disagrees with its own
inputs is a number typed by hand with extra steps.

One pareto and one calibration report per corpus. Figures are not compared:
their bytes depend on the matplotlib build, and the markdown already carries
every number they plot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalgate.commands.report import CALIBRATION_FILE, PARETO_FILE, corpora, pareto_for
from evalgate.config import load_config, resolve_path
from evalgate.pipeline import build_tokenizer
from evalgate.reporting.calibration import build_report as build_calibration
from evalgate.reporting.pareto import latest_per_config, load_summaries

# What `make report` runs with no PROFILE.
PROFILE = ["+experiment=baseline"]
STALE = "is stale; run `make report` and commit the result"
CORPORA = ["cbam_synthetic", "cbam"]


def test_every_corpus_on_disk_has_its_reports(repo_root: Path) -> None:
    cfg = load_config(overrides=PROFILE)
    summaries = latest_per_config(load_summaries(resolve_path(cfg, "paths.runs_dir")))
    assert set(corpora(resolve_path(cfg, "paths.eval_dir"), summaries)) == set(CORPORA)


@pytest.mark.parametrize("corpus", CORPORA)
def test_the_committed_pareto_report_matches_the_runs(
    repo_root: Path, tmp_path: Path, corpus: str
) -> None:
    cfg = load_config(overrides=PROFILE)
    summaries = latest_per_config(load_summaries(resolve_path(cfg, "paths.runs_dir")))
    rendered = pareto_for(summaries, corpus, tmp_path)
    name = PARETO_FILE.format(corpus=corpus)
    committed = (repo_root / "reports" / name).read_text(encoding="utf-8")
    assert committed == rendered, f"reports/{name} {STALE}"


@pytest.mark.parametrize("corpus", CORPORA)
def test_the_committed_calibration_report_matches_the_judgments(
    repo_root: Path, corpus: str
) -> None:
    cfg = load_config(overrides=[*PROFILE, f"corpus={corpus}"])
    rendered = build_calibration(cfg, build_tokenizer(cfg))
    name = CALIBRATION_FILE.format(corpus=corpus)
    committed = (repo_root / "reports" / name).read_text(encoding="utf-8")
    assert committed == rendered, f"reports/{name} {STALE}"


def test_the_calibration_report_does_not_depend_on_the_profile(repo_root: Path) -> None:
    """It describes the judgments on disk, not whichever judge is configured."""
    offline = load_config(overrides=PROFILE)
    groq = load_config(overrides=["+experiment=groq"])
    assert build_calibration(offline, build_tokenizer(offline)) == build_calibration(
        groq, build_tokenizer(groq)
    )
