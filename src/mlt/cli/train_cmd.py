"""mlt train: run a training job."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import click

from mlt.cli.discovery import require_project_dir


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
              help="Stream the training log to the terminal in real time.")
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

    from mlt.core.settings import resolve_effective_settings

    settings = resolve_effective_settings(
        project_dir,
        python_flag=python_ver,
        seed_flag=seed,
        runner=runner,
        env=os.environ,
    )

    click.echo(f"  python  {settings.python_requested}  ({settings.python_source})")
    click.echo(f"  runner  {settings.runner_display}")
    click.echo(f"  seed    {settings.seed}  ({settings.seed_source})")
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

    # --verbose: stream build.log then train.log in real time.
    # Phase 1 (build env): tail project/.mlt/build.log from the moment train()
    # is called; the trainer deletes any previous build.log before starting.
    # Phase 2 (fit):       switch to run_dir/train.log when on_run_created fires.
    def _tail_file(log_path: Path, stop: threading.Event) -> None:
        """Read log_path continuously until stop is set, then drain and exit."""
        deadline = time.monotonic() + 10.0
        while not log_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            with log_path.open(encoding="utf-8", errors="replace") as fh:
                while not stop.is_set():
                    chunk = fh.read(4096)
                    if chunk:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                    else:
                        time.sleep(0.1)
                # Drain remaining output after stop is signalled.
                remainder = fh.read()
                if remainder:
                    sys.stdout.write(remainder)
                    sys.stdout.flush()
        except OSError:
            pass  # file disappeared (e.g. run removed on contract error)

    _stop_build = threading.Event()
    _stop_train = threading.Event()
    _build_thread: threading.Thread | None = None
    _train_thread: threading.Thread | None = None

    if verbose:
        sys.stdout.write("\n--- build.log ---\n")
        sys.stdout.flush()
        _build_thread = threading.Thread(
            target=_tail_file,
            args=(project_dir / ".mlt" / "build.log", _stop_build),
            daemon=True,
        )
        _build_thread.start()

    def _on_run_created(rid: str) -> None:
        nonlocal _train_thread
        # Stop build log streaming and switch to train log.
        if verbose:
            _stop_build.set()
            if _build_thread is not None:
                _build_thread.join(timeout=2.0)
            sys.stdout.write("\n--- train.log ---\n")
            sys.stdout.flush()
            _train_thread = threading.Thread(
                target=_tail_file,
                args=(project_dir / ".mlt" / "runs" / rid / "train.log", _stop_train),
                daemon=True,
            )
            _train_thread.start()

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
        on_run_created=_on_run_created,
    )

    # Stop train log streaming and wait for drain.
    _stop_train.set()
    if _train_thread is not None:
        _train_thread.join(timeout=2.0)

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
    click.echo("Suggested steps:")
    click.echo(f"  mlt runs show {run_id}")
    click.echo(f"  mlt serve {run_id}")

