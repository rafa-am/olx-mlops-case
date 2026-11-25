# mlt

A tool for data scientists who write only Python code: from reproducible, isolated training to a REST inference API, with a single CLI.

The scientist provides `model.py`, `requirements.txt`, and data. Everything else — environment isolation, artifact storage, run tracking, and serving — is `mlt`'s responsibility.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.10 | Host interpreter used to run `mlt` itself |
| Docker (optional) | Primary isolation mode; without it, `mlt` falls back to venv automatically |
| Network access on first build | Packages are downloaded once; subsequent runs use the cache |

---

## Installation

```bash
pip install -e ".[dev]"   # development (editable, includes test deps)
pip install .             # production
```

Verify:

```bash
mlt --version
# mlt 0.1.0
# python 3.10.12
# docker: available, daemon running
```

---

## Quickstart — end to end in 6 steps

### Step 1 — Create a project from the iris template

```bash
mlt init ~/demo --template iris
cd ~/demo
```

The iris template ships with `model.py`, `requirements.txt`, `params.yaml`, `schema.yaml`, `.python-version`, and `data/iris.csv`.

### Step 2 — Validate the contract (no training yet)

```bash
mlt check .
```

```
ok  model.py     Model found (SklearnModel)
ok  requirements.txt  present
ok  schema.yaml  5 features declared
ok  data/        150 files
```

### Step 3 — Train

```bash
mlt train .
```

```
  python  3.10  (source: .python-version)
  runner  auto → docker
  seed    42  (source: default)

  building environment (may be slow the first time)...
  training...

  run_id  20261125-143000-a1b2c3
  status  succeeded
  accuracy  0.9667

  mlt runs show 20261125-143000-a1b2c3
  mlt serve 20261125-143000-a1b2c3
```

> On subsequent runs with the same inputs, `mlt train` returns immediately and prints the existing run ID — no redundant retraining.

### Step 4 — Inspect the run

```bash
mlt runs list
mlt runs show latest
```

`runs show` displays identity, inputs (code hash, params, data hash, seed, Python), environment (runner, isolation), and outputs (metrics, model path).

### Step 5 — Serve the model

```bash
mlt serve latest
```

```
  Waiting for model to load...
  Ready.

  URL      http://127.0.0.1:8000
  run_id   20261125-143000-a1b2c3
  docs     http://127.0.0.1:8000/docs

  curl -s -X POST http://127.0.0.1:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"inputs": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}'
```

> The URL is only printed after `/health` returns 200 — the model is always loaded before you can call it.

### Step 6 — Predict

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]}' \
  | python -m json.tool
```

```json
{
    "predictions": ["setosa"],
    "run_id": "20261125-143000-a1b2c3"
}
```

Swagger UI with an interactive editor is available at `http://127.0.0.1:8000/docs`.

---

## Scientist's guide

### What to provide

| File | Required | Role |
|------|----------|------|
| `model.py` | yes | one class inheriting from `SklearnModel` or `MLModel` |
| `requirements.txt` | yes | all packages `model.py` imports; pinned versions recommended |
| `data/` | yes | training data folder (or `--data <path>`) |
| `params.yaml` | no | hyperparameters passed as `params` to `fit` |
| `schema.yaml` | no | input feature types and constraints; inferred from training if absent |
| `.python-version` | no | one line, e.g. `3.12`; `requirements.txt` cannot declare Python |

### Choosing a base class

| Situation | Base class | What you implement |
|-----------|------------|--------------------|
| scikit-learn (or any joblib-compatible estimator) | `SklearnModel` | `fit` only; assign `self.model` |
| Any other framework (PyTorch, XGBoost, custom) | `MLModel` | `fit`, `predict`, `save`, `load` |

Start with the template:

```bash
mlt init <folder> --template iris   # scikit-learn example
```

### Writing `model.py`

```python
from mlt.model.sklearn import SklearnModel   # or: from mlt.model.base import MLModel

class Model(SklearnModel):

    def fit(self, data_dir, params):
        """
        data_dir: pathlib.Path — read-only folder with your training data.
        params:   dict — contents of params.yaml (empty dict if absent).
        self.seed is set by mlt before fit is called.

        Return a dict of metrics (name -> float). They appear in `runs list`
        and are compared by `mlt verify`.
        """
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(data_dir / "iris.csv")
        X, y = df.drop(columns="species"), df["species"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            random_state=self.seed,
        ).fit(X_tr, y_tr)
        return {"accuracy": float(self.model.score(X_te, y_te))}
```

`SklearnModel` provides `predict`, `save`, and `load` automatically via joblib.  
`predict` accepts features in any column order; it reorders them to match training.

### Rules

1. **`fit`** reads from `data_dir` (read-only). Use `tempfile` for temporaries. Everything printed in `fit` goes to `train.log`.
2. **`predict`** receives `list[dict]` with the same column names used in training and returns a `list` of the same length. JSON-serializable.
3. **`self.seed`** is assigned by `mlt` before `fit` runs. Pass it to every estimator and `train_test_split`.
4. **`requirements.txt`** must list everything `model.py` imports. The environment has nothing else.
5. **Data** lives in `<project>/data/` by default, or in the path given by `--data`. It is never copied into the Docker image.
6. **Schema**: if `fit` trains on a named DataFrame, `mlt` infers the schema from `feature_names_in_`. To add type constraints or required fields, write `schema.yaml`.

### `schema.yaml` (optional)

```yaml
features:
  sepal_length: {type: float, min: 0.0}
  sepal_width:  {type: float, min: 0.0}
  petal_length: {type: float, min: 0.0}
  petal_width:  {type: float, min: 0.0}
```

Supported types: `int`, `float`, `str`. Optional constraints: `min`, `max`, `allowed`, `required`.

---

## CLI reference

| Command | What it does |
|---------|-------------|
| `mlt --version` | Version, Python, Docker status |
| `mlt init <folder> --template iris` | Create a new project from the iris template |
| `mlt check [folder]` | Validate the contract without training; `--deep` builds the environment and validates inside it |
| `mlt train [folder]` | Train and record a run; flags: `--runner auto\|docker\|venv`, `--seed N`, `--timeout 600`, `--force`, `--verbose` |
| `mlt runs list` | Last 20 runs (run_id, status, duration, metrics, fingerprint); `--all` |
| `mlt runs show <run_id>` | Full run detail; `latest` and unique prefix accepted |
| `mlt diff <a> <b>` | What changed between two runs |
| `mlt verify <run_id>` | Retrain from the run's own snapshot and compare metrics |
| `mlt serve <run_id>` | Start the inference API; `--port 8000`, `--host 127.0.0.1` |

**Run ID shortcuts:** every command that takes a `run_id` also accepts `latest` and any unique prefix (like `git`).

### What `mlt` controls

| File / flag | Effect | Default |
|-------------|--------|---------|
| `--runner` | `auto` (Docker if available, else venv), `docker`, `venv` | `auto` |
| `--seed` / `MLT_SEED` | Random seed; recorded in the manifest | 42 |
| `--python` / `.python-version` | Python version for the isolated environment | interpreter running `mlt` (venv) or `3.12` (Docker) |
| `--timeout` | Training timeout in seconds | 600 |
| `--data` | Data directory | `<project>/data/` |

---

## Inference API

| Endpoint | Description |
|----------|-------------|
| `POST /predict` | `{"inputs": [{"feature": value, …}]}` → `{"predictions": […], "run_id": "…"}` |
| `GET /schema` | Effective input schema for the served run |
| `GET /metadata` | run_id, fingerprint, metrics, schema hash |
| `GET /health` | 200 when the model is loaded |
| `GET /docs` | Swagger UI with interactive example |

Error codes: 400 (malformed JSON), 413 (batch > 1000), 422 (schema violation with field path), 500 (exception in `predict`).

---

## Reproducibility and traceability

Every run records a **fingerprint** — a hash of code, params, schema, deps, Python version, and seed. If you run `mlt train` with the same inputs, `mlt` returns the existing run instead of retraining.

```bash
mlt verify <run_id>   # retrain from the run's own code/params snapshot and compare
mlt diff <a> <b>      # see exactly what changed between two runs
```

Data traceability is identity-level by default: `mlt` records the hash of each file in the data folder. Without `--snapshot-data` (P2), `verify` refuses if data has changed, pointing you to `diff`.

---

## Run storage

```
<project>/
  .mlt/
    runs/
      <run_id>/
        manifest.json      source of truth for the run
        model/             artifacts written by save()
        metrics.json
        train.log
        code/              snapshot of the project at training time
        params.yaml
        schema_effective.yaml
    index.sqlite           query index (rebuildable from manifests)

~/.mlt/
  envs/<env_hash>/         cached venv (reused across projects with same deps)
```

The SQLite index is a cache; the manifest is authoritative. A `mlt verify` retrains from `code/` inside the run, never from the project root.

---

## What was cut (P2)

The following flows are fully specified in `docs/design.md` §16 but not implemented:

- `--snapshot-data`: content-addressed data snapshot for full reproducibility without relying on stable files
- `serve --detach`, `mlt ps`, `mlt stop`, `mlt logs`: background serving with process management
- `blank` template: `MLModel` skeleton for non-sklearn frameworks
- `--json` output flag on all commands
- `runs list --status` filter
- `kill -9` reconciliation: a hard-killed training process leaves the run as `running`; a manual `mlt runs list` will not auto-recover it (Ctrl+C is handled correctly)

---

## Running the tests

```bash
# Fast loop (~20s, no Docker required)
pytest -m "not slow and not docker"

# Full suite without Docker (~8 min, builds a venv)
pytest -m "not docker"

# Docker tests (~30 min, requires Docker daemon)
pytest -m docker
```
