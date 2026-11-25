"""Training orchestrator.

``train()`` is the single entry point for running a training job.
The CLI (etapa 3) calls this function; serving and verify also use it.

Flow (matches docs/design.md §6):
  1. Static contract validation (AST)              → ContractError (exit 3)
  2. Resolve seed and Python version
  3. Load params/schema; set up DataSource; check data
  4. Calculate input hashes and fingerprint
  5. Idempotency: return existing run if fingerprint matches
  6. Build the isolated environment                → MltEnvironmentError (exit 6)
  7. Create run (status: running)
  8. Copy code/ and effective params/schema to run_dir
  9. Execute training worker                       → worker exit code
 10. On success: complete run (status: succeeded)
     On exit 3: remove run; re-raise ContractError
     On timeout: fail run; raise MltTimeoutError   (exit 5)
     On other failure: fail run; raise ExecutionError (exit 4)
"""

from __future__ import annotations

import secrets
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from mlt.core.contract import validate_contract
from mlt.core.errors import (
    ContractError,
    ExecutionError,
    MltEnvironmentError,
    MltTimeoutError,
)
from mlt.core.fingerprint import (
    build_fingerprint,
    get_git_info,
    hash_code,
    hash_data_dir,
    hash_deps,
    hash_params,
    hash_schema_file,
    resolve_python,
)
from mlt.core.seed import resolve_seed
from mlt.datasources.local import LocalDataSource
from mlt.runners.venv import VenvRunner
from mlt.runners.venv import _TIMEOUT_EXIT_CODE
from mlt.runstore.manifest import Manifest
from mlt.runstore.store import RunStore


def train(
    project_dir: Path,
    data_dir: Path | None = None,
    runner: str = "auto",
    python: str | None = None,
    seed: int | None = None,
    timeout: int = 600,
    force: bool = False,
    env: Mapping[str, str] | None = None,
    on_slow_build: "Callable[[], None] | None" = None,
    verification_of: str | None = None,
    _store_project_dir: Path | None = None,
) -> str:
    """Run a training job and return the run_id.

    Parameters
    ----------
    project_dir:  Root of the scientist's project (must contain model.py).
    data_dir:     Data directory; defaults to ``project_dir/data/``.
    runner:       ``"auto"`` (venv; Docker if available), ``"venv"``, or ``"docker"``.
    python:       Requested Python version (``"3.10"``); overrides ``.python-version``.
    seed:         Random seed; overrides MLT_SEED env var.
    timeout:      Training timeout in seconds.
    force:        Re-train even if an identical run already exists.
    env:          Environment mapping for seed resolution (defaults to os.environ).
    on_slow_build: Called when the environment build is known to be slow.
    """
    import os

    if env is None:
        env = os.environ

    project_dir = Path(project_dir).resolve()
    store_project_dir = Path(_store_project_dir).resolve() if _store_project_dir else project_dir

    # 1. Static validation -------------------------------------------------
    validate_contract(project_dir)

    # 2. Resolve seed and Python -------------------------------------------
    seed_val, seed_origin = resolve_seed(seed, env)
    python_minor, python_requested, python_source = resolve_python(project_dir, python)

    # Venv runner enforces that python_minor matches sys.executable.
    _check_python_compat(python_minor, python_requested, python_source, runner)

    # 3. Load params and check data ----------------------------------------
    params = _load_params(project_dir)
    effective_data_dir = data_dir if data_dir is not None else project_dir / "data"
    datasource = LocalDataSource(effective_data_dir)
    datasource.check()

    # 4. Calculate input hashes and fingerprint ----------------------------
    data_name = Path(effective_data_dir).name
    # Exclude data dir name only if it is a direct child of project_dir.
    extra_exclude: set[str] = set()
    if Path(effective_data_dir).parent.resolve() == project_dir:
        extra_exclude.add(data_name)

    code_h = hash_code(project_dir, extra_exclude=extra_exclude)
    params_h = hash_params(params)
    schema_h = hash_schema_file(project_dir / "schema.yaml")
    deps_h = hash_deps(project_dir / "requirements.txt")
    data_info = datasource.hash_files()
    git_commit, dirty = get_git_info(project_dir)
    fingerprint = build_fingerprint(
        code_h, params_h, schema_h, deps_h, python_minor, seed_val
    )

    # 5. Idempotency -------------------------------------------------------
    store = RunStore(store_project_dir)
    if not force:
        existing = store.find_by_fingerprint(fingerprint)
        if existing is not None:
            return existing.run_id

    # 6. Build environment -------------------------------------------------
    req_txt = project_dir / "requirements.txt"
    runner_instance = _create_runner(runner, req_txt, python_minor)
    lock_str = runner_instance.build_env(on_slow_build=on_slow_build)
    deps_lock = [line for line in lock_str.splitlines() if line.strip()]
    eh = runner_instance.get_env_hash()

    # 7. Create run --------------------------------------------------------
    run_id = _new_run_id()
    partial = Manifest.new(
        run_id=run_id,
        inputs=_build_inputs(
            code_h, git_commit, dirty,
            params_h, params,
            schema_h, project_dir,
            python_requested, python_source,
            data_info,
            deps_h,
            seed_val, seed_origin,
        ),
        fingerprint=fingerprint,
        environment={
            "runner": runner_instance.name,
            "python": sys.version.split()[0],
            "isolation": runner_instance.isolation,
            "env_hash": eh,
        },
        verification_of=verification_of,
    )
    run_dir = store.create_run(partial)

    # 8. Copy project snapshot and effective files -------------------------
    _copy_code(project_dir, run_dir / "code", exclude_names=extra_exclude | {".mlt"})
    _copy_if_exists(project_dir / "params.yaml", run_dir / "params.yaml")
    _copy_if_exists(project_dir / "schema.yaml", run_dir / "schema.yaml")

    # 9. Execute training --------------------------------------------------
    try:
        exit_code = runner_instance.run_training(
            run_dir=run_dir,
            project_dir=project_dir,
            data_dir=effective_data_dir,
            seed=seed_val,
            timeout=timeout,
        )
    except KeyboardInterrupt:
        store.fail_run(run_id, "interrupted")
        raise

    # 10. Handle result ----------------------------------------------------
    if exit_code == 0:
        metrics = _read_metrics(run_dir)
        schema_outputs = _read_schema_outputs(run_dir, project_dir)
        artifacts = _list_artifacts(run_dir / "model")
        partial.mark_succeeded(
            metrics=metrics,
            schema_outputs=schema_outputs,
            artifacts=artifacts,
            deps_lock=deps_lock,
            env_hash=eh,
        )
        store.complete_run(partial)
        return run_id

    if exit_code == _TIMEOUT_EXIT_CODE:
        store.fail_run(run_id, "timeout")
        raise MltTimeoutError(
            f"training timed out after {timeout}s; "
            f"see {run_dir / 'train.log'}"
        )

    if exit_code == 3:
        # Contract error detected inside the environment: clean up the run
        # so it doesn't appear in `runs list` as a failed run.
        store.remove_run(run_id)
        log_excerpt = _tail_log(run_dir / "train.log")
        raise ContractError(
            f"deep validation failed inside the environment:\n{log_excerpt}"
        )

    store.fail_run(run_id, f"worker exited with code {exit_code}")
    raise ExecutionError(
        f"training failed (exit {exit_code}); "
        f"see {run_dir / 'train.log'}"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_python_compat(
    python_minor: str,
    python_requested: str,
    python_source: str,
    runner: str,
) -> None:
    """Raise MltEnvironmentError if the venv runner can't satisfy the Python request."""
    import sys

    running_minor = ".".join(str(v) for v in sys.version_info[:2])
    if python_minor == running_minor:
        return  # compatible

    if runner in ("docker",):
        return  # Docker can use any Python; venv cannot

    # auto or venv: cannot use a different Python
    raise MltEnvironmentError(
        f"Python {python_requested} requested (via {python_source}) "
        f"but the venv runner uses Python {running_minor} (the interpreter "
        f"running mlt).  Use --runner docker or install mlt in a venv with "
        f"Python {python_requested}."
    )


def _load_params(project_dir: Path) -> dict:
    params_path = project_dir / "params.yaml"
    if not params_path.exists():
        return {}
    import yaml

    raw = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _create_runner(runner_name: str, req_txt: Path, python_minor: str):
    """Instantiate the right runner based on the requested name."""
    from mlt.core.docker_check import docker_available
    from mlt.runners.docker import DockerRunner

    if runner_name == "docker":
        if not docker_available():
            raise MltEnvironmentError(
                "Docker is not available or the daemon is not running. "
                "Use --runner venv or start the Docker daemon."
            )
        return DockerRunner(req_txt, python_minor)

    if runner_name == "auto":
        if docker_available():
            return DockerRunner(req_txt, python_minor)
        import click
        click.echo(
            "  warning: Docker not available, falling back to venv (isolation: none)",
            err=True,
        )
        return VenvRunner(req_txt, python_minor)

    # explicit venv
    return VenvRunner(req_txt, python_minor)


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(3)}"


def _build_inputs(
    code_h, git_commit, dirty,
    params_h, params,
    schema_h, project_dir,
    python_requested, python_source,
    data_info,
    deps_h,
    seed_val, seed_origin,
) -> dict:
    schema_source = "declared" if (project_dir / "schema.yaml").exists() else "none"
    return {
        "code": {"hash": code_h, "git_commit": git_commit, "dirty": dirty},
        "params": {"hash": params_h, "values": params},
        "schema": {"hash": schema_h, "source": schema_source},
        "python": {"requested": python_requested, "source": python_source},
        "data": {
            "hash": data_info["hash"],
            "source": "local",
            "files": data_info["files"],
            "snapshot": None,
            "frozen": False,
        },
        "deps": {"hash": deps_h, "lock": []},  # lock filled in after build
        "seed": seed_val,
        "seed_source": seed_origin,
    }


def _copy_code(src: Path, dst: Path, exclude_names: set[str]) -> None:
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(*exclude_names, "__pycache__", "*.pyc"),
    )


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _read_metrics(run_dir: Path) -> dict:
    import json

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {}


def _read_schema_outputs(run_dir: Path, project_dir: Path) -> dict:
    import hashlib

    effective = run_dir / "schema_effective.yaml"
    declared = project_dir / "schema.yaml"

    if effective.exists():
        content = effective.read_bytes()
        h = hashlib.sha256(content).hexdigest()
        source = "declared" if declared.exists() else "inferred"
        return {"hash": h, "source": source}
    return {"hash": None, "source": "none"}


def _list_artifacts(model_dir: Path) -> dict[str, str]:
    import hashlib

    artifacts: dict[str, str] = {}
    if not model_dir.exists():
        return artifacts
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(model_dir.parent))
            artifacts[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return artifacts


def _tail_log(log_path: Path, lines: int = 20) -> str:
    if not log_path.exists():
        return "(no log)"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    tail = text.strip().splitlines()[-lines:]
    return "\n".join(tail)
