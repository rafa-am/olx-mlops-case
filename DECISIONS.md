# DECISIONS.md: MLOps Case

## Overview

`mlt` (**M**odel **L**ifecycle **T**oolkit) is a Python package with a CLI that takes the data scientist from reproducible training to a REST API, from Python code alone.

The scientist provides `requirements.txt`, `model.py` (a class that inherits from `MLModel`), data and, optionally, `params.yaml` and `schema.yaml`. The tool handles isolation, tracking, and serving.

1. `mlt init` creates a project from a template (Iris is the example).
2. `mlt check` validates the contract without training.
3. `mlt train` runs training in Docker (primary) or venv, without network, and records a run with the model, metrics, and manifest.
4. `mlt runs list` and `mlt runs show` identify each run.
5. `mlt serve` starts a FastAPI for a specific run.

The CLI is a thin layer over `mlt.core`. `DataSource`, `Runner`, and `RunStore` are interfaces with local implementation in the MVP and a path of evolution toward the cloud.

## Scientist's contract

The minimum contract is a folder with `requirements.txt` and `model.py`. `params.yaml`, `schema.yaml`, and data are optional or external.

- `model.py` defines exactly one subclass of `MLModel`, with `fit(data_dir, params)` (returns a dict of metrics) and `predict(inputs)` (list of records as input, list of predictions as output).
- `save` and `load` are also abstract in the core. The optional `SklearnModel` class provides a default `predict` and `save`/`load` via joblib; the minimum a scientist writes is `fit`, assigning `self.model`. The iris template also shows `predict`, `save`, and `load` explicitly (delegating to `super()`) so scientists understand what each method does and when to override.
- The seed arrives in `self.seed`; the scientist uses it in estimators.
- `params.yaml`, if it exists, becomes the `params` argument. `schema.yaml`, if it exists, declares the input features; without it, the schema is inferred from `feature_names_in_`.

**Python version:** `requirements.txt` cannot declare it, so there is an optional `.python-version` file (one line, pyenv and uv convention). Precedence: `--python`, then the file, then the default.

**Why a class and not a function:** a common base provides reuse, a single point for verifying the contract, and a guaranteed `predict`. A function would be simpler for short scripts, but would scatter the conventions.

**Why the core is framework-agnostic:** a default `predict` based on scikit-learn Pipeline in the core would couple the tool to one artifact type and make others difficult, such as PySpark-trained models. The core requires `fit`, `predict`, `save`, and `load`, and scikit-learn ergonomics come in `SklearnModel`, optionally. The discarded alternative was putting the default `predict` and joblib in `MLModel`.

**Known limit:** `fit(data_dir, params)` assumes data in a local folder. A Spark model would need a `DataSource` that provides a URI and a cluster `Runner`. The tool could pass to `fit` only the parameters it declares (like pytest fixtures), but this is not implemented in the MVP.

**Why schema in YAML and not in Pydantic in the class:** removes the definition from code, allows reviewing and versioning what was inferred, and keeps the contract smaller. Loses cross-field validations, which is recorded as a limitation.

## Isolation and reproducibility

Each training run executes in an isolated environment identified by hashes, and the run records everything that identifies it.

- **Docker (primary):** image `mlt-env:<hash>`, with hash of `requirements.txt`, Python version, and `mlt` version. The base is `python:3.12-slim` (`--python` changes it). The build has network access; training runs without network, with read-only data.
- **venv (alternative):** same contract, with cache in `~/.mlt/envs/<hash>`. Does not isolate network or system, and the manifest records the isolation level.
- **Fingerprint:** hash of code, data, params, declared schema, dependencies, and seed. It is the identity of the run.
- **Seed:** `--seed`, then `MLT_SEED`, then 42. The effective value and origin go in the manifest.
- **Dependencies:** after the build, `pip freeze` is recorded as a lock in the manifest. Requirements without pinned versions generate a warning.
- **Guarantee:** same fingerprint means same identified inputs. `verify` compares metrics with tolerance, because the seed does not eliminate non-determinism from threads and libraries.

**Known limit:** the `mlt` runtime is installed in the same environment as the scientist's dependencies, so pinned versions may conflict with FastAPI or Pydantic. Isolating the runtime in two environments was left as future work.

**How `mlt` enters the image:** the runner copies the installed package to the build context. Since the version does not change during development, the environment hash also includes a hash of the `mlt` source code, and `--rebuild` forces reconstruction.

**Network and mounts:** the build needs network access once (packages and base image); after that, with cache, training runs offline. In the container, project and data are read-only, the run output is writable, temporaries go in `tmpfs`, and the user is not root.

**Python and runner:** in Docker, the base image comes from the requested version. In venv, Python is the interpreter that runs `mlt` (`sys.executable`), without searching for other versions in the system; a different version than requested exits with code 6, explaining how to fix it. The major.minor version enters the fingerprint, and the runner stays outside it (informational only), so the same project in Docker and in venv has the same fingerprint. Without Docker, `auto` falls back to venv with a warning, and the manifest records `isolation: none`.

**`verify` and `diff`:** `diff` compares two manifests and explains why the fingerprints differ. `verify` is the practical proof of reproducibility: retrains only from what the run saved (`code/`, params, schema, seed, and Python), creates a run with `verification_of`, and compares metrics with a fixed relative tolerance of 1e-6 (exit 0 if reproduced, 4 if diverged). It does not prove equal data content without a snapshot, nor bit-for-bit reproduction across machines. If time is short, the command can be cut and the same argument lives in the integration test that trains twice.

## Data

Data stays outside the image and enters training through a `DataSource` interface, so the origin can change without altering the contract.

- **MVP:** local folder mounted as a read-only volume (`/data:ro`), without copying. `fit` receives `data_dir`.
- **Evolution:** S3 or lakehouse (Iceberg, Parquet), materialized outside the sandbox, so training continues without network. For volumes that do not fit on disk, direct reading via VPC endpoint.
- **Traceability in levels:** identify (hash per file and aggregate, always), verify (`mlt verify` refuses to retrain if data changed), and freeze (`--snapshot-data`, content-addressed copy).
- **Without freezing**, the guarantee is of identity, not content: you can say whether two runs used the same data, but not recover it. With a lakehouse, the immutable snapshot resolves this without copying.

Column statistics in the manifest would help diagnose changes, but only identify them; left out of the MVP.

**Data and `check`:** `check` asks the `DataSource` whether the source is usable (in the MVP, the local folder exists and is not empty; in a lakehouse it would be a metadata query). In training, the `DataSource` materializes the data into a local read-only directory, cached outside the project and outside the image. So the snapshot does not need to exist in the scientist's folder.

## Run, manifest, and RunStore

Each training run generates an immutable run in `<project>/.mlt/runs/<run_id>/`, with model, `metrics.json`, `manifest.json`, effective `params.yaml` and `schema.yaml`, `code/` (copy of the project without `data/` and `.mlt/`), and `train.log`. The run is self-contained: `verify` retrains from `code/`.

- **Manifest (JSON):** blocks `inputs` (enter the fingerprint), `fingerprint`, `environment`, `outputs`, and run metadata, with `manifest_version`.
- **SQLite** is only a query index (identity, fingerprint, environment, metrics, paths). The manifest is the source of truth, and the index is rebuildable.
- **Root is input, run is record:** training copies `params.yaml` and `schema.yaml` to the run. `serve`, `verify`, and `diff` read from the run; `check` reads from the root.
- **Project anchor:** `.mlt/` lives inside the project (recognized by `model.py`) and is ignored in the code hash. Other commands discover the project by walking up directories, like Git.
- **Concurrency:** `run_id` with random suffix and exclusive directory creation, atomic manifest write, SQLite in WAL mode, and environment lock during build. Runs stuck in `running` without a live process become `failed` (`interrupted`).
- **Failures:** contract errors do not create a run. Execution errors and timeouts create a `failed` run, with `error` and preserved log.
- **Registry readiness:** the run is a self-contained package, and serving always obtains it via `RunStore`.

## Inference API and serving

The API serves a specific run and has four endpoints, with a minimal contract and explicit errors.

- **`POST /predict`:** input `{"inputs": [records]}`, output `{"predictions": [...], "run_id": "..."}`. One prediction per record, in the same order, free-form and JSON-serializable. Validation is all-or-nothing: 422 (index and field), 400, 413, 500, or 503.
- **`GET /schema`, `/metadata`, and `/health`:** effective schema of the run, identification of the served model, and readiness.
- **Lifecycle:** `mlt serve <run_id>` in the foreground (or `--detach`), `ps`, `stop`, and `logs`. The alias `latest` resolves to the concrete `run_id` on startup. Only serves `succeeded` runs, and listens on 127.0.0.1.
- **Image:** training environment plus the generic server. The model is mounted read-only.

**CLI exit codes:** 0 success, 1 unexpected error, 2 incorrect usage, 3 contract, 4 execution failure, 5 timeout, 6 environment unavailable, 7 not found, 8 operation refused by state.

Supplements: 130 for Ctrl+C interruption (the run becomes `failed`, `interrupted`); 4 also for `verify` that diverges; 6 also for a busy port and for a Python version unavailable in venv.

## Implementation decisions (step 2)

**Run created before the worker runs — exit 3 deletes it, does not mark it failed.**
The design says "contract error inside the environment does not create a run". In practice the run is created in `running` before launching the subprocess (to guarantee consistency in case of Ctrl+C). If the worker exits with code 3 (ContractError in deep validation), the trainer calls `store.remove_run()` and deletes the directory — visible result: no run in `runs list`. If the worker exits with code 4 (ExecutionError in fit), the run is marked `failed` and kept (log preserved). This preserves the design invariant without requiring two subprocesses (validate-only + train).

**Worker as subpackage `mlt._worker` (installed in the venv).**
An alternative would be to copy a standalone script to the build context. Using an installed subpackage alongside `mlt` is cleaner: no temporary file, no absolute path passed to the subprocess, and any change in the worker is covered by the `mlt_source_hash` that invalidates the venv cache.

**VenvRunner uses `sys.executable` — no PATH search.**
If `.python-version` or `--python` request a different version than the interpreter running `mlt`, the trainer raises `MltEnvironmentError` (exit 6) with a clear instruction: use Docker or reinstall `mlt` in a venv with that version. The alternative (looking for `python3.X` in PATH) would make behavior dependent on the host environment and hard to test.

## Implementation decisions (step 3)

**Error handling in the CLI — `_MltGroup.invoke` instead of `main()`.**
The initial approach captured `MltError` only in `main()`. In tests via `CliRunner` that invoke `cli` directly, the error was not caught by Click and `result.exit_code` came as 1 (generic exception) instead of the correct code. The solution was to create a subclass `_MltGroup(click.Group)` that overrides `invoke` and catches `MltError`, translating to `ctx.exit(code)`. This way, the correct exit code is returned both in real execution (via `main()` in standalone_mode=True) and in tests via CliRunner.

**Project discovery in two modes.**
`train` and `check` receive the path explicitly and require `model.py` (error 2 if absent). `runs`, `diff`, and `verify` walk up directories from the cwd until finding `model.py` or `.mlt/`, with fallback to cwd. This distinction reflects the design: run management commands work from inside the project (like `git`), while `train` and `check` need an explicit target to avoid accidentally training the wrong project.

**`verify` stores the verification run in the original project's store, not in `code/`.**
`verify` trains from `code/` (snapshot of the original run), but storing the run in `code/.mlt/` would make the verification invisible in the project's `mlt runs list`. The solution was to add the `_store_project_dir` parameter to `trainer.train()`: the RunStore is initialized with the original project, and the verification run appears in the listing with the `[V]` marker and with `verification_of` filled in the manifest. The parameter is prefixed with `_` to signal internal use.

**`verify` refuses if the data hash changed — exit 8.**
Without `--snapshot-data`, data reproducibility is by identity. If the current hash diverges from the original manifest, `verify` refuses with `OperationRefusedError` (exit 8) and directs the user to use `mlt diff` or `--snapshot-data`. An alternative would be to train anyway and warn — but that would produce a run with different data, which contradicts the goal of verify.

**`iris.csv` embedded in the iris template.**
The iris template had no example data, which broke the promise of `mlt init` ("next steps: cd, check, train") — the scientist had to generate the CSV manually before being able to train. The decision was to include `data/iris.csv` (150 rows, ~6 KB, public domain) directly in `src/mlt/templates/iris/data/`. `mlt init` already copies everything via `template_dir.rglob("*")`, with no code changes. Alternative considered: a `generate_data.py` script in the template, more explicit but with more friction. The embedded CSV was preferred for simplicity.

**`--rebuild` deferred.**
`--rebuild` forces reconstruction of the cached venv environment. The implementation would require changes to `VenvRunner` (deleting the env directory and recreating it) plus the flag in the CLI. Since it is an operational edge case (corrupted environment) that does not affect correctness, it was deferred. `--force` covers the most common case (retraining ignoring idempotency).

## Implementation decisions (v0.2.0 — core isolation)

**Core extracted from CLI: no click/typer in `mlt.core`.**
`runs_cmd`, `diff_cmd`, `verify_cmd`, `train_cmd`, and `discovery.py` previously contained business logic mixed with formatting. The extraction moved each piece to a pure function in `mlt.core` returning a typed dataclass (`RunDiff`, `VerifyResult`, `EffectiveSettings`), leaving the CLI as a thin formatter. This is the prerequisite for the Web UI, which calls the same core without going through Click. The boundary is enforced by `test_core_isolation.py`. The alternative (keeping logic in cmd modules and duplicating it in the UI) was discarded to avoid drift between CLI and UI behavior.

**`verify_run` returns `VerifyResult` on divergence instead of raising.**
An earlier draft raised `ExecutionError` when metrics diverged, attaching the result to the exception. This was replaced by always returning `VerifyResult(reproduced=False, ...)` and letting the CLI raise `ExecutionError` when it needs exit code 4. Reasons: (1) it avoids modifying the exception hierarchy to carry structured data, (2) the UI can inspect `reproduced` without catching exceptions, (3) it makes the core easier to test without try/except.

**`VerifyResult` carries fingerprint hashes explicitly.**
To format the divergence message (`fp_a[:8] ≠ fp_b[:8]`), the CLI originally re-fetched the manifests from the store after `verify_run` returned. This was replaced by storing `orig_fingerprint` and `new_fingerprint` directly in `VerifyResult`, eliminating the extra store reads and making the result self-contained.

**`iris/model.load` bug: missing `@classmethod` and `return`.**
The iris template defined `load` as an instance method (`def load(self, path)`), shadowing the `@classmethod` from `SklearnModel`. `server.py` calls `model_cls.load(path)`, which passes `path` as `self` and fails with `TypeError`. Fixed by adding `@classmethod` and `return super().load(path)`. The same error appeared in the README and `docs/design.md` examples, corrected in the same commit. A regression test (`test_iris_template_load_is_classmethod`) was added to `test_model.py`.

## Implementation decisions (test suite overhaul)

**`_force_venv` autouse fixture — non-docker tests never use Docker.**
Before this fixture, `runner="auto"` with Docker available caused all 142 non-docker tests to build and run Docker containers, blurring the boundary between the two test profiles and making the suite non-deterministic across machines. The fixture monkeypatches `mlt.core.docker_check.docker_available` to return `False` for any test without the `docker` marker. This keeps the non-docker suite fast and deterministic regardless of the host environment.

**Three-tier test suite — `not slow`, `slow`, `docker`.**
Tests that call `train()` or build a venv are marked `pytest.mark.slow` at the module level. Tests that require the Docker daemon keep `pytest.mark.docker`. Everything else (`contract`, `fingerprint`, `model`, `seed`, etc.) runs in seconds with no external dependencies. The three profiles are: `pytest -m "not slow and not docker"` (~104 tests, seconds), `pytest -m "slow and not docker"` (39 tests, ~8 min with warm cache), and `pytest -m docker` (9 tests, ~30 min).

**Unified requirements across all slow tests — single venv.**
`minimal_project` and `iris_project` use the same `requirements.txt` content (`scikit-learn>=1.3 + pandas>=2.0 + numpy>=1.24`). The venv cache key is `env_hash(deps, python_minor, mlt_source_hash)`, so a single venv is built and reused across all 39 slow tests. The "bad model" tests (`test_check_deep_venv_fails_bad_model`, `test_docker_runner_deep_check_fails_bad_model`) also use these requirements — the contract error is detected before any package is imported, so the extra dependencies are irrelevant. Same logic for Docker: all 9 tests share one image. Alternative: keep minimal requirements for minimal fixtures — rejected because the savings in cold-build time outweigh the semantic impurity.

**Lazy imports in iris template `model.py`.**
The original template imported `pandas` and `sklearn` at module level. When `--validate-only` runs inside Docker, it imports `model.py` to check the contract — triggering sklearn/numpy import before `fit` is called. On some CPU configurations this caused SIGILL (exit 132) on the first container run, while later tests passed (numpy already warm). Moving imports inside `fit()` aligns with the lazy-import rule already applied to `SklearnModel`, eliminates the cold-start crash, and reduces container startup time for deep checks.

## Implementation decisions (step 6)

**KeyboardInterrupt mapping in `_MltGroup.invoke`:** `trainer.py` already catches `KeyboardInterrupt`, marks the run as `failed` (error: `"interrupted"`), and re-raises. The gap was that `_MltGroup.invoke` only caught `MltError`; in a CliRunner test (and in some shell configurations) the re-raised `KeyboardInterrupt` would not be translated to exit code 130. Fix: add `except KeyboardInterrupt: ctx.exit(130)` to `_MltGroup.invoke`. This is the minimal change — no new logic in the core.

**Ctrl+C slow test omitted:** testing a real `KeyboardInterrupt` with a live subprocess would require sending `SIGINT` from a background thread while `runner.run_training()` blocks. The added complexity and flakiness (timing-sensitive) are not justified when the fast test already covers the exit-code mapping, and the actual `except KeyboardInterrupt` path in `trainer.py` is exercised by manual end-to-end verification.

**Two-tier test design for failures:** fast tests (no venv) mock `mlt.core.trainer.train` and exercise only the CLI exit-code mapping. Slow tests create real venvs and verify run-store state after each failure. The separation keeps the fast loop under 25 seconds while still having end-to-end coverage for the most critical failure paths.

**`mlt.core.trainer.train` as the monkeypatch target:** `train_cmd` imports `train` lazily (inside the function body), so `mlt.cli.train_cmd.train` does not exist as a module attribute. Monkeypatching the source module (`mlt.core.trainer.train`) works because the lazy import binds at call time and picks up the patched object.

## Implementation decisions (step 5)

**FastAPI app as a factory with injected model — `create_app(manifest, schema, model)`.**
The serving module exposes a factory that receives the already-loaded model, rather than loading it internally. This allows tests to inject a mock model (pure Python, no sklearn) and use `TestClient` without starting a subprocess, venv, or Docker container. In production, `server.py` loads the model from the run directory and passes it to the factory. The alternative — loading inside `create_app` — would have required a real run directory for every test, coupling the unit tests to the filesystem.

**Server runs inside the isolated environment (venv or Docker), not the host.**
`mlt serve` launches `python -m mlt.serving.server <run_dir>` inside the same environment used for training (venv at `~/.mlt/envs/<env_hash>/` or Docker image `mlt-env:<hash>`). This is required because loading the model needs the scientist's dependencies (joblib, sklearn, etc.), which only exist in the isolated environment. The host Python does not have them. The serve command reads `manifest.environment.env_hash` to find the right environment.

**`_mlt_pkg.py` — shared utility for editable vs. non-editable mlt installs.**
Both VenvRunner and DockerRunner need to install mlt inside the isolated environment. With an editable install (`pip install -e .`), `mlt.__file__` points to the source tree and `pip install -e <root>` works. With a non-editable install (`pip install mlt`), the fallback in `_find_mlt_root()` returned `site-packages/`, which caused `shutil.copytree` to try to copy the entire site-packages directory. The fix: `find_mlt_source_root()` returns `None` when no source tree is found, and `write_mlt_installable()` reconstructs a minimal pip-installable layout from the installed package plus runtime deps read from `importlib.metadata`. This avoids hardcoding the dep list in a second place.

**`SklearnModel.predict` reorders columns by `feature_names_in_`.**
sklearn >= 1.0 raises `ValueError` if a DataFrame passed to `predict` has column names in a different order than training. Since `pd.DataFrame(list_of_dicts)` preserves insertion order (which may differ from the training order), predictions would fail for any client that sends features in an order other than the exact training order. The fix uses `model.feature_names_in_` (set automatically by sklearn when trained with a named DataFrame) to reorder the columns before calling `predict`. If `feature_names_in_` is absent (array input at training time), the input order is used unchanged.

**`openapi_extra` for the `/predict` body in Swagger.**
The `/predict` endpoint uses `Request` as its parameter type to retain manual control over error codes (400/413/422/500 per the design). FastAPI cannot infer the request body schema from `Request`, so the Swagger UI did not show the body editor. The fix adds `openapi_extra` with a JSON Schema description and an example record generated from the run's schema. This gives Swagger full documentation without changing the error handling logic.

**`init` accepts any dot-prefixed entry, not a named whitelist.**
The original check used `_SAFE = {".venv", ".git", ".DS_Store"}` and blocked any other hidden entry. Any real project directory (with `.python-version`, `.mypy_cache`, etc.) would fail even before running `mlt init`. The rule was simplified to `name.startswith(".")`: hidden entries coexist, visible entries signal an occupied project directory.

## Implementation decisions (Web UI — v0.3.0)

**FastAPI SPA with hash routing, single HTML file.**
The UI is a single `index.html` served by the same FastAPI app that exposes the JSON API. Navigation is hash-based (`#/train`, `#/verify/<id>`) so the server never needs to handle frontend routes. Alternative: a separate frontend build (Vite/React) — rejected because it adds a build step and a second process to a local-only tool that already has FastAPI as a dependency.

**Thread-safe state objects (`_TrainState`, `_VerifyState`, `_ServeState`) instead of shared dicts.**
Each long-running operation runs in a background thread and writes to a small state object behind a `threading.Lock`. The API endpoints only read snapshots (`snapshot() → dict`). The alternative (asyncio) would have required either running blocking `train()` in a thread pool anyway, or making the entire core async — not worth it for a local-only UI with one user.

**`on_step` and `on_run_created` callbacks in `trainer.train()`.**
The UI needs to update the progress display incrementally: step names as they start, and the new run_id as soon as the run record exists (before training completes). Both are injected as optional callbacks with `None` defaults, so the CLI and tests are unaffected. The `on_run_created` callback in particular lets the verify page switch from `build.log` to `train.log` as soon as the run record is created, without polling the store.

**`/api/system` as single source of truth for UI defaults.**
Runner names, default port, default timeout, and step list are module-level constants in `app.py`, exported via `GET /api/system`. The frontend never hardcodes these values. Any future addition to `_TRAIN_STEPS` requires a matching `_step()` call in `trainer.py` (enforced by a cross-reference comment in both files) but is otherwise automatic in the UI.

**Verify build log reads from the snapshot directory, not the project root.**
`verify_run()` calls `train()` with `project_dir=run_dir/code/`, so `build.log` is written to `run_dir/code/.mlt/build.log`, not to the live project's `.mlt/`. A dedicated `GET /api/verify/log` endpoint reads from the right path; the generic `GET /api/build/log` is left for the train flow only.

**Dismiss pattern for terminal states (train and verify).**
A `POST /api/<op>/dismiss` endpoint resets the in-memory state to idle. Without it, a cancelled or failed card persists across page navigations because the state is in memory, not derived from the run store. The alternative (derive from the store on every render) would re-show cancelled runs every time the user navigates back, which is unexpected.

## Ambiguities and interpretations

When the problem statement left something open, I chose the simplest interpretation that keeps the tool useful end-to-end.

| Point | Chosen interpretation | Reason |
| --- | --- | --- |
| Tool format | Python package with CLI, using Docker for isolation | The user provides only Python; the rest is the tool's responsibility |
| Minimum contract | `MLModel` class with `fit`, `predict`, `save`, and `load`; `SklearnModel` optional | Single and verifiable convention |
| Isolation | Docker as primary, venv as alternative | Does not depend on Docker on the evaluator's machine |
| Inference API | One service per run, without authentication | Abstracts the backend; the rest is evolution |
| Artifacts and metrics associated with the execution | Run folder with manifest, and SQLite index | Traceable without an external service |
| Public dataset and simple model | Iris with RandomForest | The focus is the tool |
| Production architecture | AWS design, not implemented | It is optional in the problem statement |

**"Only Python code":** I interpreted this as "no infrastructure or server", not as "no convention at all". The problem statement asks me to define the minimum contract and justify it, so some structure is expected. The required items are `requirements.txt` and `model.py`; `params.yaml` and `schema.yaml` are optional. They are not Python because configuration read without executing code is easier to review, version, and compare across runs.

**Installing `mlt` on the host:** `mlt` can be installed in a venv inside the project (`python3 -m venv .venv && pip install mlt`) or in a global venv shared across projects. The `.gitignore` generated by `mlt init` includes `.venv/` for the most common case (local venv), and the README mentions the global alternative. What matters is that `mlt` is on the PATH — the venv location is irrelevant to the tool.

**Adoption friction:** inheriting from `MLModel` couples `model.py` to the tool, and an existing script needs to be adapted. Mitigations: `mlt init` delivers the skeleton, the class lives in a subpackage without dependencies, and remains testable outside the tool.

**Scientist's guide:** the contract needs to be visible to tool users, not just to its designers. The README provides a one-page guide (which base class to choose, what `fit` and `predict` receive, full `requirements.txt`, data and schema), and `init` offers the `iris` (scikit-learn) and `blank` (`MLModel` skeleton) templates. Data lives in `<project>/data/` by default, or in another folder indicated by `--data`. Errors inside `predict` only appear when serving, because `check` does not train; this is stated in the guide.

**Two-step contract validation:** `check` is static and does not import `model.py`, because the scientist's dependencies only exist in the isolated environment, not on the host. Deep validation (import, instantiate, and check the methods) runs inside the environment, without network, before creating the run. This way, contract errors still do not create a run.

**`save` and `load` on a local directory:** the model writes and reads a directory created by the tool. Who transports the package between disk and a remote registry is `RunStore`, so the model never talks to the registry and the scientist's contract does not change when `RunStore` changes. The limit is that very large models or ones that write directly to a remote destination would need a URI, which fits the named-parameter injection rule.

## Alternatives considered

Each alternative below was evaluated and discarded, with the reason.

| Alternative | Why it was discarded |
| --- | --- |
| Contract by function (`train()`) | Would scatter conventions; the class provides reuse and a single verification point |
| Schema as a Pydantic attribute in the class | Keeps the definition in code; the YAML allows reviewing what was inferred |
| MLflow for run tracking | Too heavy for the scope; the `RunStore` interface allows adopting it later |
| Data embedded in the image | Heavy image and no cache; the read-only volume solves it |
| CLI as a client of a remote service from the start | That is a platform, and the problem statement does not ask for it; the separate core allows evolution |
| `.mlt/` in the current directory | The result would depend on where the command is executed; anchored to the project |
| Runtime isolated in two environments | Two processes and a protocol; deferred, with `MLModel` already in a subpackage without dependencies |
| Seed only in `params.yaml` | Would depend on the scientist declaring it; flag and environment variable guarantee the record |
| Data statistics in the manifest | Only identify, do not reproduce; outside the MVP |

## Out of scope and evolution

The MVP delivers the full local workflow; what follows was left out by choice.

**Out of scope:** authentication, data versioning, rollback and A/B, monitoring and drift, GPU, distributed training, CPU and memory limits, old run cleanup, and API route versioning.

**Delivered as v0.3.0 (was originally out of scope):** local Web UI (`mlt ui`) — read-only views (home, runs, diff) plus train, serve, and verify with live progress. The core/CLI contract was not changed; the UI calls the same `mlt.core` functions used by the CLI.

**Evolution, in the order I would do it:**

1. Isolate the `mlt` runtime from the scientist's environment (dependency-free SDK plus a worker process).
2. Remote `DataSource` (S3 and lakehouse, with snapshot).
3. CLI as a client of a control API, with `Runner` on AWS Batch or SageMaker Training.
4. `RunStore` on S3 and DynamoDB and, later, a run and model registry with aliases and promotion.
5. Optional output block in `schema.yaml` and support for `predict_proba`.

**Also as evolution:** reducing adoption friction by accepting a scikit-learn estimator or a simple training function through an adapter, without requiring `MLModel` inheritance.

**PySpark support:** a `SparkModel` with native `save` and `load`, `fit` with named-parameter injection, a cluster `Runner`, and a `DataSource` by URI.

**Delivery priorities (7 days):** P0 is what cannot be missing: contract, training in Docker and venv, manifest and RunStore, `runs list/show`, `serve` in the foreground with `/predict`, tests, README, and documents. P1: `check`, `init` with `iris` template, `verify`, `diff`. P2, cut first if time runs short: `--snapshot-data`, `serve --detach`, `ps`, `stop`, `logs`, interrupted run reconciliation, `blank` template, `--json`, and `runs list --status` (CLI flag). `serve --detach` is P2 because, without `stop`, a background service would have no way to be stopped. The flows for all P2 items are closed in `design.md`. Of these, interrupted run reconciliation was implemented (`core/reconcile.py`, called by `mlt ui` on startup); the rest remain unimplemented.

## Time spent and what I would do differently

Approximate time spent per step (generation + review + correction rounds):

| Step | What was built | Time |
|------|---------------|------|
| 1 | Package skeleton, MLModel, contract, errors, exit codes | ~25 min |
| 2 | venv training, DataSource, RunStore, fingerprint, seed | ~15 min + 8 min test run |
| 3 | Full CLI: init, check, train, runs, diff, verify | ~25 min + 12 min test run |
| 3 post | iris.csv bundled in template | ~10 min |
| Translation | All CLI output and docs to English | ~20 min |
| 4 | DockerRunner, check --deep | ~30 min |
| Test suite overhaul | Markers, unified venv, lazy iris imports | ~30 min |
| 5 | FastAPI serving, mlt serve, 6 bug fixes | ~1 h |
| 6 | Failure-mode tests, KeyboardInterrupt fix | ~15 min |
| 7 | README, docs/architecture.md | ~30 min |
| Web UI | FastAPI SPA: home, runs, train, serve, verify, diff, new project | ~6 h |

**Total: approximately 10–11 hours of active work** (excluding test execution time for slow/docker suites).

**What I would do differently with more time:**

- **Isolate the `mlt` runtime from the scientist's environment.** Today `mlt` (FastAPI, uvicorn, pydantic) is installed in the same venv or Docker image as the scientist's dependencies. This was the right call for the MVP — one environment, simple to reason about — but it creates a known version-conflict risk. With more time, I would run the server in a separate process with its own environment, keeping the scientist's environment clean.

- **Deliver the `blank` template.** It is P2 and small (~30 min). It completes the story for scientists not using scikit-learn and makes `MLModel` more tangible than just a base class described in the README.

- **Add an end-to-end subprocess test for `mlt serve`.** The serving tests use `TestClient` (no subprocess). A test that actually spawns `uvicorn` via `serve_cmd` would give stronger confidence in the full `serve_cmd → subprocess → uvicorn → model.predict` path.

- **Add UI tests.** The Web UI has no automated tests beyond the existing core/CLI suite. The main risk is frontend regressions (broken polling, stale state after navigation). A lightweight approach: a handful of `httpx` tests against `create_ui_app()` for the API endpoints, plus a few Playwright smoke tests for the most critical flows (start train, poll to done, navigate away).

- **Support switching projects without restarting.** `create_ui_app(project_dir)` binds the server to a single project at startup; opening a different project requires restarting `mlt ui`. An "Open project" button was removed from the UI for this reason. The fix would make `project_dir` a session-level setting (stored in `.mlt/ui.state`) and add a project-picker endpoint, or run one server per project with a lightweight launcher that proxies between them.

**On feature engineering and scope:**

`mlt` covers the cycle from ready data to trained model to REST API. Feature engineering — building the input dataset from raw sources — is deliberately out of scope. For more complex projects, a Feature Store (Feast, Tecton, SageMaker Feature Store) would sit upstream of `mlt`: the Feature Store delivers features, `mlt` trains and serves the model.

The architecture already has the right extension points: `DataSource` for training-time feature retrieval and a middleware layer for online enrichment at `/predict` time. But incorporating feature engineering into `mlt` itself would break its core contract — *"the scientist provides data, mlt handles the rest"* — and push it toward a full MLOps platform, which is a different tool with a different scope.
