"""Serving subprocess entry point.

Invoked by the runner as:
  python -m mlt.serving.server <run_dir> [--host H] [--port N]

Must run INSIDE the isolated environment (venv or Docker) so the
scientist's dependencies are available for model loading.

Loading flow:
  1. Add run_dir/code/ to sys.path.
  2. Import model.py and find the MLModel subclass.
  3. Call ModelClass.load(run_dir/model/).
  4. Create the FastAPI app and hand it to uvicorn.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import sys
from pathlib import Path


def _load_model(run_dir: Path):
    """Load the scientist's model from run_dir/code/model.py + run_dir/model/."""
    code_dir = run_dir / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    spec = importlib.util.spec_from_file_location("scientist_model", code_dir / "model.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    from mlt.model.base import MLModel

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, MLModel)
        and obj is not MLModel
        and obj.__module__ == module.__name__
    ]
    # Remove SklearnModel itself in case it leaked into the module namespace.
    try:
        from mlt.model.sklearn import SklearnModel
        candidates = [c for c in candidates if c is not SklearnModel]
    except ImportError:
        pass

    if not candidates:
        print("error: no MLModel subclass found in run code", file=sys.stderr)
        sys.exit(4)

    model_cls = candidates[0]
    return model_cls.load(run_dir / "model")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlt.serving.server")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    run_dir: Path = args.run_dir.resolve()

    from mlt.runstore.manifest import Manifest
    manifest = Manifest.read(run_dir / "manifest.json")

    # Effective schema is written by the training worker (may not exist).
    schema = None
    schema_path = run_dir / "schema_effective.yaml"
    if schema_path.exists():
        import yaml
        schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))

    model = _load_model(run_dir)

    import uvicorn

    from mlt.serving.app import create_app
    app = create_app(manifest, schema, model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
