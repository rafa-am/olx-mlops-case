"""Tests for the CLI entry point."""

from click.testing import CliRunner

import mlt
from mlt.cli.main import cli


def test_version_output_contains_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"mlt {mlt.__version__}" in result.output


def test_version_output_contains_python():
    import sys

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    version_str = sys.version.split()[0]
    assert f"python {version_str}" in result.output


def test_version_output_contains_docker_line():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "docker:" in result.output


def test_no_args_shows_help():
    runner = CliRunner()
    result = runner.invoke(cli, [])
    # Click returns exit code 0 for --help / no-subcommand group
    assert result.exit_code == 0
    assert "Usage:" in result.output
