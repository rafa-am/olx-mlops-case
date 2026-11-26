"""mlt verify: re-train from a run's snapshot and check reproducibility."""

from __future__ import annotations

import os
from pathlib import Path

import click

from mlt.cli.discovery import find_project_dir
from mlt.core.errors import ExecutionError, OperationRefusedError


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
    from mlt.datasources.local import LocalDataSource
    from mlt.runstore.store import RunStore

    project_dir = find_project_dir()
    store = RunStore(project_dir)
    orig = store.get_run(run_id)

    if orig.status != "succeeded":
        raise OperationRefusedError(
            f"run {run_id[:8]} has status '{orig.status}'; "
            "only succeeded runs can be verified."
        )

    # Determine effective data dir.
    effective_data = Path(data_dir).resolve() if data_dir else project_dir / "data"

    # Check data hash — refuse if data changed since the original run.
    orig_data_hash = (orig.inputs.get("data") or {}).get("hash")
    try:
        current_info = LocalDataSource(effective_data).hash_files()
    except Exception as exc:
        raise OperationRefusedError(
            f"could not read data directory {effective_data}: {exc}"
        ) from exc

    if orig_data_hash and current_info["hash"] != orig_data_hash:
        raise OperationRefusedError(
            f"data hash changed since run {run_id[:8]}. "
            "Use --snapshot-data when training to freeze data, "
            "or use mlt diff to inspect the change."
        )

    # Train from the run's code snapshot, storing the new run in the same project.
    code_dir = store.run_dir(run_id) / "code"
    if not code_dir.is_dir():
        raise OperationRefusedError(
            f"run {run_id[:8]} has no code snapshot; cannot verify."
        )

    effective_runner = runner or orig.env_runner or "auto"
    orig_python = (orig.inputs.get("python") or {}).get("requested")
    orig_seed = orig.inputs.get("seed")

    click.echo(f"  retraining from {run_id[:8]}...")
    click.echo(f"  runner  {effective_runner}  python  {orig_python or 'default'}"
               f"  seed  {orig_seed}")
    click.echo()

    from mlt.core.trainer import train

    new_run_id = train(
        project_dir=code_dir,
        data_dir=effective_data,
        runner=effective_runner,
        python=orig_python,
        seed=orig_seed,
        force=True,  # always create a new run for the verification
        env=os.environ,
        verification_of=run_id,
        _store_project_dir=project_dir,
    )

    new_m = store.get_run(new_run_id)

    # Compare fingerprints.
    fp_match = orig.fingerprint == new_m.fingerprint

    # Compare metrics with relative tolerance 1e-6.
    metric_diffs: list[str] = []
    for key, v_orig in (orig.metrics or {}).items():
        v_new = (new_m.metrics or {}).get(key)
        if v_new is None:
            metric_diffs.append(f"{key}: missing in new run")
            continue
        tol = 1e-6 * max(abs(v_orig), 1.0)
        if abs(v_orig - v_new) > tol:
            metric_diffs.append(f"{key}: {v_orig:.6g} ≠ {v_new:.6g}")

    if fp_match and not metric_diffs:
        click.echo(click.style("✓ reproducible", fg="green"))
        click.echo(f"  run_id  {new_run_id}")
    else:
        lines = []
        if not fp_match:
            lines.append(
                f"fingerprint: {(orig.fingerprint or '')[:8]} ≠ {(new_m.fingerprint or '')[:8]}"
            )
        lines.extend(metric_diffs)
        click.echo(click.style("✗ diverged", fg="red"))
        for line in lines:
            click.echo(f"  {line}")
        raise ExecutionError(
            f"verification of {run_id[:8]} diverged: "
            + "; ".join(lines)
        )
