"""mlt train: run a training job."""

from __future__ import annotations

import os
from pathlib import Path

import click

from mlt.cli.discovery import require_project_dir
from mlt.core.errors import UsageError


@click.command("train")
@click.argument("directory", default=".", required=False)
@click.option("--data", "data_dir", default=None, metavar="PATH",
              help="Data directory (default: <project>/data/).")
@click.option(
    "--runner",
    default="auto",
    show_default=True,
    type=click.Choice(["auto", "venv", "docker"]),
    help="Isolation runner.",
)
@click.option("--python", "python_ver", default=None, metavar="VERSION",
              help="Python version (e.g. 3.10); overrides .python-version.")
@click.option("--seed", default=None, type=int,
              help="Random seed; overrides MLT_SEED env var.")
@click.option("--timeout", default=600, show_default=True, type=int,
              help="Training timeout in seconds.")
@click.option("--force", is_flag=True, default=False,
              help="Re-train even if an identical run already exists.")
@click.option("--verbose", is_flag=True, default=False,
              help="Print the training log to the terminal after training.")
def train_cmd(
    directory: str,
    data_dir: str | None,
    runner: str,
    python_ver: str | None,
    seed: int | None,
    timeout: int,
    force: bool,
    verbose: bool,
) -> None:
    """Train a model and record a run."""
    project_dir = require_project_dir(directory)
    effective_data = Path(data_dir).resolve() if data_dir else project_dir / "data"

    # Resolve effective values to show the preamble before training starts.
    from mlt.core.fingerprint import resolve_python
    from mlt.core.seed import resolve_seed

    python_minor, python_requested, python_source = resolve_python(project_dir, python_ver)
    seed_val, seed_origin = resolve_seed(seed, os.environ)

    from mlt.core.docker_check import docker_available

    if runner == "auto":
        effective_runner = "docker" if docker_available() else "venv"
        runner_display = f"auto → {effective_runner}"
    else:
        runner_display = runner

    click.echo(f"  python  {python_requested}  ({python_source})")
    click.echo(f"  runner  {runner_display}")
    click.echo(f"  seed    {seed_val}  ({seed_origin})")
    click.echo()

    # Callback printed when the env build is detected as slow (first time).
    _build_noted = [False]

    def _on_slow_build() -> None:
        if not _build_noted[0]:
            _build_noted[0] = True
            click.echo(
                "  building environment (may be slow the first time)...",
                err=False,
            )

    click.echo("  training...")

    from mlt.core.trainer import train

    run_id = train(
        project_dir=project_dir,
        data_dir=effective_data,
        runner=runner,
        python=python_ver,
        seed=seed,
        timeout=timeout,
        force=force,
        env=os.environ,
        on_slow_build=_on_slow_build,
    )

    # Load the manifest to show the summary.
    from mlt.runstore.store import RunStore

    store = RunStore(project_dir)
    m = store.get_run(run_id)

    click.echo()
    status_str = (
        click.style("succeeded", fg="green")
        if m.status == "succeeded"
        else click.style(m.status, fg="red")
    )
    click.echo(f"  run_id  {run_id}")
    click.echo(f"  status  {status_str}")

    if m.metrics:
        for key, val in m.metrics.items():
            click.echo(f"  {key:<8}  {val:.4f}")

    click.echo()
    click.echo(f"  mlt runs show {run_id}")
    click.echo(f"  mlt serve {run_id}")

    if verbose:
        log_path = store.run_dir(run_id) / "train.log"
        if log_path.exists():
            click.echo()
            click.echo("--- train.log ---")
            click.echo(log_path.read_text(encoding="utf-8", errors="replace"))
