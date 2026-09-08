from __future__ import annotations

import pytest

from evalgate.cli import COMMANDS, main


def test_help_lists_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in COMMANDS:
        assert name in out


def test_no_args_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: evalgate" in capsys.readouterr().out


def test_unknown_command_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["nope"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_config_command_runs(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config"]) == 0
    assert "config_hash" in capsys.readouterr().out


def test_config_command_accepts_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "retriever=bm25"]) == 0
    assert "bm25" in capsys.readouterr().out
