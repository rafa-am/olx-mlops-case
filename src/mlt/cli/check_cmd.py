"""mlt check: static contract validation."""

from __future__ import annotations

from pathlib import Path

import click

from mlt.cli.discovery import require_project_dir
from mlt.core.contract import CheckResult, Level, check_project
from mlt.core.errors import ContractError


def _icon(level: Level) -> str:
    if level == Level.OK:
        return click.style("✓", fg="green")
    if level == Level.WARNING:
        return click.style("⚠", fg="yellow")
    return click.style("✗", fg="red")


def _print_results(results: list[CheckResult]) -> bool:
    """Print check results; return True if any error was found."""
    has_error = False
    for r in results:
        loc = ""
        if r.file:
            loc = r.file
            if r.line:
                loc = f"{loc}:{r.line}"
            loc = f"  [{loc}]"
        click.echo(f"  {_icon(r.level)} {r.message}{loc}")
        if r.level == Level.ERROR:
            has_error = True
        if r.suggestion and r.level != Level.OK:
            click.echo(f"      → {r.suggestion}")
    return has_error


@click.command("check")
@click.argument("directory", default=".", required=False)
@click.option(
    "--data",
    "data_dir",
    default=None,
    metavar="PATH",
    help="Data directory (default: <project>/data/).",
)
@click.option(
    "--deep",
    is_flag=True,
    default=False,
    help="Build the environment and validate inside it (not yet implemented).",
)
def check_cmd(directory: str, data_dir: str | None, deep: bool) -> None:
    """Validate the project contract without training."""
    project_dir = require_project_dir(directory)
    effective_data = Path(data_dir).resolve() if data_dir else project_dir / "data"

    # Show effective python for context.
    from mlt.core.fingerprint import resolve_python

    python_minor, python_requested, python_source = resolve_python(project_dir, None)
    click.echo(f"  python  {python_requested}  ({python_source})")
    click.echo(f"  data    {effective_data}")
    click.echo()

    # Static contract checks.
    results = check_project(project_dir)

    # Data check via DataSource.
    from mlt.datasources.local import LocalDataSource

    try:
        LocalDataSource(effective_data).check()
        results.append(
            CheckResult(
                level=Level.OK,
                message="data directory found and not empty",
                file=str(effective_data),
            )
        )
    except ContractError as exc:
        results.append(
            CheckResult(
                level=Level.ERROR,
                message=exc.message,
                file=str(effective_data),
                suggestion="Create the data directory and add at least one data file.",
            )
        )

    has_error = _print_results(results)

    if deep:
        click.echo()
        click.echo(click.style("  ⚠ --deep not yet implemented", fg="yellow"))

    click.echo()
    if has_error:
        click.echo(click.style("error", fg="red"))
        raise ContractError("one or more contract checks failed")
    else:
        click.echo(click.style("ok", fg="green"))
