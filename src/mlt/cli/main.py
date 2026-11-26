"""CLI entry point.

The CLI is a thin layer: it parses arguments, calls mlt.core, and translates
MltError exceptions into the correct exit codes. All logic lives in the core.

``_MltGroup`` overrides ``invoke`` so that MltError is caught and translated
to the right exit code whether the CLI is called via ``main()`` (production) or
directly via Click's CliRunner (tests).
"""

import subprocess
import sys

import click

import mlt
from mlt.core.errors import MltError


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


class _MltGroup(click.Group):
    """Click Group that maps MltError exceptions to their correct exit codes.

    Overriding ``invoke`` means the mapping works in both production (via
    ``main()``) and in tests that call ``cli`` through Click's CliRunner.
    """

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except MltError as exc:
            click.echo(f"error: {exc.message}", err=True)
            ctx.exit(int(exc.exit_code))
        except KeyboardInterrupt:
            ctx.exit(130)


@click.group(cls=_MltGroup, invoke_without_command=True)
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


# Register subcommands.
from mlt.cli.check_cmd import check_cmd  # noqa: E402
from mlt.cli.diff_cmd import diff_cmd  # noqa: E402
from mlt.cli.init_cmd import init_cmd  # noqa: E402
from mlt.cli.runs_cmd import runs_cmd  # noqa: E402
from mlt.cli.serve_cmd import serve_cmd  # noqa: E402
from mlt.cli.train_cmd import train_cmd  # noqa: E402
from mlt.cli.verify_cmd import verify_cmd  # noqa: E402

cli.add_command(init_cmd)
cli.add_command(check_cmd)
cli.add_command(train_cmd)
cli.add_command(runs_cmd)
cli.add_command(diff_cmd)
cli.add_command(verify_cmd)
cli.add_command(serve_cmd)


def main() -> None:
    """Entry point registered in pyproject.toml."""
    cli(standalone_mode=True)
