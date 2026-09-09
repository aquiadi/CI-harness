from __future__ import annotations

from pathlib import Path

from scripts.render_readme import build_values, render


def test_the_committed_readme_matches_the_current_artifacts(repo_root: Path) -> None:
    """A stale number in the README fails the build rather than misleading a reader."""
    committed = (repo_root / "README.md").read_text(encoding="utf-8")
    assert committed == render(repo_root), "README.md is stale; run `make readme`"


def test_no_placeholder_survives_rendering(repo_root: Path) -> None:
    assert "${" not in render(repo_root)


def test_headline_values_come_from_the_baseline(repo_root: Path) -> None:
    values = build_values(repo_root)
    assert float(values["quality"]) > 0.0
    assert float(values["p95_ms"]) > 0.0
    assert values["corpus"] == "cbam_synthetic"
    assert values["n_configs"].isdigit()
    assert int(values["n_configs"]) >= 12


def test_the_readme_keeps_the_deliberately_skipped_section(repo_root: Path) -> None:
    """This section is deliberate and must not be quietly dropped."""
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "## Tooling we deliberately skipped" in readme
    assert "No DVC. No MLflow." in readme


def test_the_readme_states_what_the_numbers_were_measured_on(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "synthetic" in readme
    assert "## Limitations" in readme
