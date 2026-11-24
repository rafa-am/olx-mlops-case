"""CLI entry point.

The CLI is a thin layer: it parses arguments, calls mlt.core, and translates
MltError exceptions into the correct exit codes. All logic lives in the core.
"""

import subprocess
import sys

import click

import mlt
from mlt.core.errors import ExitCode, MltError


def _detect_docker() -> str:
    """Return a human-readable string describing Docker availability.

    Calls ``docker info`` with a short timeout so the version output is never
    delayed by a hanging daemon check.
    """
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "available, daemon running"
        return "available, daemon not running"
    except FileNotFoundError:
        return "unavailable"
    except subprocess.TimeoutExpired:
        return "available, daemon not responding"


def _version_callback(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"mlt {mlt.__version__}")
    click.echo(f"python {sys.version.split()[0]}")
    click.echo(f"docker: {_detect_docker()}")
    ctx.exit()


@click.group(invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_version_callback,
    help="Show version, Python and Docker status, then exit.",
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """mlt: from reproducible training to a REST API, from Python alone."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def main() -> None:
    """Entry point that maps MltError exceptions to exit codes."""
    try:
        cli(standalone_mode=False)
    except MltError as exc:
        click.echo(f"error: {exc.message}", err=True)
        sys.exit(int(exc.exit_code))
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except click.exceptions.Exit as exc:
        sys.exit(exc.code)
    except click.exceptions.Abort:
        click.echo("Aborted.", err=True)
        sys.exit(int(ExitCode.INTERRUPTED))
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(int(ExitCode.UNEXPECTED_ERROR))
