"""Training worker: runs INSIDE the isolated environment (venv or Docker).

Invoked by the runner as:
  python -m mlt._worker.train \
    --project-dir <path> --data-dir <path> --run-dir <path> --seed <int>

Exit codes (must match mlt.core.errors.ExitCode):
  0  success
  3  contract error (deep validation failed; caller deletes the run)
  4  execution error (fit raised; caller marks the run as failed)
  1  unexpected error
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Deep validation
# ---------------------------------------------------------------------------


def _load_model_class(project_dir: Path):
    """Import model.py and return the single MLModel subclass."""
    spec = importlib.util.spec_from_file_location(
        "scientist_model", project_dir / "model.py"
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    # Find subclasses of MLModel or SklearnModel defined in the module.
    from mlt.model.base import MLModel

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, MLModel) and obj is not MLModel and obj.__module__ == module.__name__
    ]

    # Try importing SklearnModel too so indirect subclasses of it are captured.
    try:
        from mlt.model.sklearn import SklearnModel

        candidates = [
            obj
            for obj in candidates
            if issubclass(obj, MLModel)  # already filtered above; keep list
        ]
        # Remove SklearnModel itself if it somehow appeared.
        candidates = [c for c in candidates if c is not SklearnModel]
    except ImportError:
        pass

    if len(candidates) == 0:
        _exit_contract("no MLModel subclass found in model.py")
    if len(candidates) > 1:
        names = ", ".join(c.__name__ for c in candidates)
        _exit_contract(f"model.py must define exactly one MLModel subclass, found: {names}")

    cls = candidates[0]

    # Verify instantiation works (no __init__ args required).
    try:
        instance = cls()
    except TypeError as exc:
        _exit_contract(f"class {cls.__name__} cannot be instantiated without arguments: {exc}")

    # Verify required methods are callable.
    for method in ("fit", "predict", "save", "load"):
        if not callable(getattr(instance, method, None)):
            _exit_contract(f"class {cls.__name__} is missing callable method '{method}'")

    return cls


def _exit_contract(msg: str) -> None:
    print(f"contract error: {msg}", file=sys.stderr)
    sys.exit(3)


def _exit_execution(msg: str) -> None:
    print(f"execution error: {msg}", file=sys.stderr)
    sys.exit(4)


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------


def _infer_schema(model_instance) -> dict | None:
    """Try to infer features from sklearn's feature_names_in_."""
    inner = getattr(model_instance, "model", None)
    if inner is None:
        return None
    names = getattr(inner, "feature_names_in_", None)
    if names is None:
        return None
    features = {str(name): {"type": "float"} for name in names}
    return {"features": features}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlt._worker.train")
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    project_dir: Path = args.project_dir
    data_dir: Path = args.data_dir
    run_dir: Path = args.run_dir
    seed: int = args.seed

    # --- Deep validation (inside the environment) -------------------------
    model_cls = _load_model_class(project_dir)

    # --- Apply seed -------------------------------------------------------
    from mlt.core.seed import apply_seed

    apply_seed(seed)

    # --- Load params ------------------------------------------------------
    params: dict = {}
    params_file = run_dir / "params.yaml"
    if params_file.exists():
        import yaml

        raw = yaml.safe_load(params_file.read_text(encoding="utf-8"))
        params = raw if isinstance(raw, dict) else {}

    # --- Fit --------------------------------------------------------------
    model = model_cls()
    model.seed = seed

    try:
        metrics = model.fit(data_dir, params)
    except Exception as exc:
        _exit_execution(f"fit raised {type(exc).__name__}: {exc}")

    if not isinstance(metrics, dict):
        _exit_execution(
            f"fit must return dict[str, float], got {type(metrics).__name__}"
        )
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            _exit_execution(
                f"fit metric '{k}' must be a number, got {type(v).__name__}"
            )

    # --- Save model -------------------------------------------------------
    model_dir = run_dir / "model"
    model_dir.mkdir(exist_ok=True)
    try:
        model.save(model_dir)
    except Exception as exc:
        _exit_execution(f"save raised {type(exc).__name__}: {exc}")

    # --- Write metrics.json -----------------------------------------------
    (run_dir / "metrics.json").write_text(
        json.dumps({k: float(v) for k, v in metrics.items()}, indent=2),
        encoding="utf-8",
    )

    # --- Schema inference / copy ------------------------------------------
    schema_declared = run_dir / "schema.yaml"
    schema_effective_path = run_dir / "schema_effective.yaml"

    if schema_declared.exists():
        import shutil

        shutil.copy(schema_declared, schema_effective_path)
    else:
        inferred = _infer_schema(model)
        if inferred is not None:
            import yaml

            schema_effective_path.write_text(
                yaml.dump(inferred, default_flow_style=False), encoding="utf-8"
            )

    print(f"training complete: {json.dumps({'metrics': dict(metrics)})}")
    sys.exit(0)


if __name__ == "__main__":
    main()
