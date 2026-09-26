"""The committed reports are what the committed artifacts render to.

The README has had this guarantee since D-0035. The reports did not, and
`reports/pareto.md` sat at 63 configurations while `runs/` held 65 and the
README -- rendered from the same runs -- said 65. A report that disagrees with
its own inputs is a number typed by hand with extra steps.

Figures are not compared: their bytes depend on the matplotlib build, and the
markdown already carries every number they plot.
"""

from __future__ import annotations

from pathlib import Path

from evalgate.config import load_config, resolve_path
from evalgate.pipeline import build_tokenizer
from evalgate.reporting.calibration import build_report as build_calibration
from evalgate.reporting.pareto import build_report as build_pareto
from evalgate.reporting.pareto import latest_per_config, load_summaries, partition_comparable

# What `make report` runs with no PROFILE.
PROFILE = ["+experiment=baseline"]
STALE = "is stale; run `make report` and commit the result"


def test_the_committed_pareto_report_matches_the_runs(repo_root: Path, tmp_path: Path) -> None:
    cfg = load_config(overrides=PROFILE)
    comparable, excluded = partition_comparable(
        latest_per_config(load_summaries(resolve_path(cfg, "paths.runs_dir")))
    )
    rendered = build_pareto(comparable, excluded, tmp_path)
    committed = (repo_root / "reports" / "pareto.md").read_text(encoding="utf-8")
    assert committed == rendered, f"reports/pareto.md {STALE}"


def test_the_committed_calibration_report_matches_the_judgments(repo_root: Path) -> None:
    cfg = load_config(overrides=PROFILE)
    rendered = build_calibration(cfg, build_tokenizer(cfg))
    committed = (repo_root / "reports" / "judge_calibration.md").read_text(encoding="utf-8")
    assert committed == rendered, f"reports/judge_calibration.md {STALE}"


def test_the_calibration_report_does_not_depend_on_the_profile(repo_root: Path) -> None:
    """It describes the judgments on disk, not whichever judge is configured."""
    offline = load_config(overrides=PROFILE)
    groq = load_config(overrides=["+experiment=groq"])
    assert build_calibration(offline, build_tokenizer(offline)) == build_calibration(
        groq, build_tokenizer(groq)
    )
