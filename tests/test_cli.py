from __future__ import annotations

from typer.testing import CliRunner

from pytorch_transformers.cli import app


def test_cli_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("train", "eval", "translate", "prepare-data", "export"):
        assert command in result.output