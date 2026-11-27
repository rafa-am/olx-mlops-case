"""Core logic for verifying run reproducibility.

Returns a structured VerifyResult; never prints.
Raises typed errors (OperationRefusedError) on bad input (wrong status,
data changed, no code snapshot). Never raises on divergence — the caller
inspects ``result.reproduced`` and maps it to an exit code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class VerifyResult:
    """Structured result of a verify operation."""

    reproduced: bool
    new_run_id: str
    fingerprint_match: bool
    # Human-readable descriptions of each metric divergence.
    metric_diffs: list[str] = field(default_factory=list)

    # Fingerprint hashes for divergence display (populated when fp_match=False).
    orig_fingerprint: str | None = None
    new_fingerprint: str | None = None

    # Effective settings used for the re-train (informational).
    effective_runner: str = ""
    effective_python: str | None = None
    effective_seed: int | None = None


def verify_run(
    project_dir: Path,
    run_id: str,
    data_dir: Path | None = None,
    runner: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_runner_created: Callable | None = None,
    on_step: Callable[[str], None] | None = None,
    on_run_created: Callable[[str], None] | None = None,
) -> VerifyResult:
    """Re-train from a run's snapshot and check reproducibility.

    Parameters
    ----------
    project_dir:  Root of the scientist's project.
    run_id:       Run to verify (accepts prefix and 'latest').
    data_dir:     Data directory; defaults to ``project_dir/data/``.
    runner:       Override runner; defaults to the original run's runner.
    on_progress:  Optional callback called with progress lines (no newline).
                  The CLI passes ``click.echo``; the UI can display them live.

    Returns
    -------
    VerifyResult.  ``reproduced=True`` when fingerprint and metrics match.
    The caller is responsible for mapping ``reproduced=False`` to an exit code.

    Raises
    ------
    OperationRefusedError  Run not succeeded, data changed, or no code snapshot.
    ExecutionError         Re-training itself failed.
    NotFoundError          run_id not found (exit 7).
    """
    from mlt.core.errors import OperationRefusedError
    from mlt.core.trainer import train
    from mlt.datasources.local import LocalDataSource
    from mlt.runstore.store import RunStore

    store = RunStore(project_dir)
    orig = store.get_run(run_id)

    if orig.status != "succeeded":
        raise OperationRefusedError(
            f"run {run_id[:8]} has status '{orig.status}'; "
            "only succeeded runs can be verified."
        )

    effective_data = data_dir or project_dir / "data"

    # Refuse if data changed since the original run.
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

    code_dir = store.run_dir(orig.run_id) / "code"
    if not code_dir.is_dir():
        raise OperationRefusedError(
            f"run {run_id[:8]} has no code snapshot; cannot verify."
        )

    effective_runner = runner or orig.env_runner or "auto"
    orig_python = (orig.inputs.get("python") or {}).get("requested")
    orig_seed = orig.inputs.get("seed")

    if on_progress:
        on_progress(f"  retraining from {orig.run_id[:8]}...")
        on_progress(
            f"  runner  {effective_runner}"
            f"  python  {orig_python or 'default'}"
            f"  seed  {orig_seed}"
        )
        on_progress("")

    new_run_id = train(
        project_dir=code_dir,
        data_dir=effective_data,
        runner=effective_runner,
        python=orig_python,
        seed=orig_seed,
        force=True,
        env=os.environ,
        verification_of=orig.run_id,
        _store_project_dir=project_dir,
        on_runner_created=on_runner_created,
        on_step=on_step,
        on_run_created=on_run_created,
    )

    new_m = store.get_run(new_run_id)

    fp_match = orig.fingerprint == new_m.fingerprint

    metric_diffs: list[str] = []
    for key, v_orig in (orig.metrics or {}).items():
        v_new = (new_m.metrics or {}).get(key)
        if v_new is None:
            metric_diffs.append(f"{key}: missing in new run")
            continue
        tol = 1e-6 * max(abs(v_orig), 1.0)
        if abs(v_orig - v_new) > tol:
            metric_diffs.append(f"{key}: {v_orig:.6g} ≠ {v_new:.6g}")

    return VerifyResult(
        reproduced=fp_match and not metric_diffs,
        new_run_id=new_run_id,
        fingerprint_match=fp_match,
        metric_diffs=metric_diffs,
        orig_fingerprint=orig.fingerprint,
        new_fingerprint=new_m.fingerprint,
        effective_runner=effective_runner,
        effective_python=orig_python,
        effective_seed=orig_seed,
    )
