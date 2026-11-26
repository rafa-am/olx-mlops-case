"""mlt verify: re-train from a run's snapshot and check reproducibility."""

from __future__ import annotations

from pathlib import Path

import click

from mlt.cli.discovery import find_project_dir
from mlt.core.errors import ExecutionError


@click.command("verify")
@click.argument("run_id")
@click.option(
    "--runner",
    default=None,
    type=click.Choice(["auto", "venv", "docker"]),
    help="Override the runner (default: same as original run).",
)
@click.option("--data", "data_dir", default=None, metavar="PATH",
              help="Data directory (default: <project>/data/).")
def verify_cmd(run_id: str, runner: str | None, data_dir: str | None) -> None:
    """Re-train from a run's snapshot and verify reproducibility."""
    from mlt.core.verify import verify_run

    project_dir = find_project_dir()
    effective_data = Path(data_dir).resolve() if data_dir else None

    result = verify_run(
        project_dir=project_dir,
        run_id=run_id,
        data_dir=effective_data,
        runner=runner,
        on_progress=click.echo,
    )

    if result.reproduced:
        click.echo(click.style("✓ reproducible", fg="green"))
        click.echo(f"  run_id  {result.new_run_id}")
    else:
        lines = []
        if not result.fingerprint_match:
            orig_fp = (result.orig_fingerprint or "")[:8]
            new_fp = (result.new_fingerprint or "")[:8]
            lines.append(f"fingerprint: {orig_fp} ≠ {new_fp}")
        lines.extend(result.metric_diffs)
        click.echo(click.style("✗ diverged", fg="red"))
        for line in lines:
            click.echo(f"  {line}")
        raise ExecutionError(
            f"verification of {run_id[:8]} diverged: " + "; ".join(lines)
        )
