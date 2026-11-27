"""mlt verify: re-train from a run's snapshot and check reproducibility."""

from __future__ import annotations

import sys
import threading
import time
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
@click.option("--verbose", is_flag=True, default=False,
              help="Stream the build and training logs to the terminal in real time.")
def verify_cmd(
    run_id: str,
    runner: str | None,
    data_dir: str | None,
    verbose: bool,
) -> None:
    """Re-train from a run's snapshot and verify reproducibility."""
    from mlt.core.verify import verify_run
    from mlt.runstore.store import RunStore

    project_dir = find_project_dir()
    effective_data = Path(data_dir).resolve() if data_dir else None

    # Derive the snapshot code_dir so we can tail its build.log.
    # verify_run() uses the same path internally.
    store = RunStore(project_dir)
    orig = store.get_run(run_id)
    code_dir = store.run_dir(orig.run_id) / "code"

    def _tail_file(log_path: Path, stop: threading.Event) -> None:
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
                remainder = fh.read()
                if remainder:
                    sys.stdout.write(remainder)
                    sys.stdout.flush()
        except OSError:
            pass

    _stop_build = threading.Event()
    _stop_train = threading.Event()
    _build_thread: threading.Thread | None = None
    _train_thread: threading.Thread | None = None

    if verbose:
        sys.stdout.write("\n--- build.log ---\n")
        sys.stdout.flush()
        _build_thread = threading.Thread(
            target=_tail_file,
            args=(code_dir / ".mlt" / "build.log", _stop_build),
            daemon=True,
        )
        _build_thread.start()

    def _on_run_created(rid: str) -> None:
        nonlocal _train_thread
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

    result = verify_run(
        project_dir=project_dir,
        run_id=run_id,
        data_dir=effective_data,
        runner=runner,
        on_progress=click.echo,
        on_run_created=_on_run_created if verbose else None,
    )

    _stop_train.set()
    if _train_thread is not None:
        _train_thread.join(timeout=2.0)

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
