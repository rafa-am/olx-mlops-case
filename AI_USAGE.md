# AI_USAGE.md: MLOps Case

## Tools used

I used Claude in conversation to plan the design before writing code, and collaborative whiteboards to record each approved decision.

- **Claude (chat):** raising alternatives and trade-offs, detailing contracts, drafting proposals.
- **Whiteboards:** keeping the design live (flow, contracts, data, manifest, API, tests).
- **Implementation:** Claude Code (CLI) generated the code for step 1 under the author's review.

## How AI entered the process

The design was done in iterative conversation, point by point, before any implementation.

Each point followed the same cycle:

1. The AI proposed a solution with alternatives and trade-offs.
2. I questioned, chose, or adjusted.
3. The decision was recorded on the whiteboard and, later, in the documents.

The order was: reading the problem statement, initial plan, scientist's contract, data and traceability, manifest, API and serving, demo, seed and failures, repository structure, tests, and environment.

## Flows and prompts that worked

What yielded the most was asking for alternatives with trade-offs before deciding, and asking direct questions about guarantees.

- **Questions about limits:** "In the MVP, how do I have a reproducibility guarantee for a run if the downloaded data changes?" led to the separation between identity-level and content-level traceability.
- **Scope questions:** "This might increase complexity; is it worth addressing now or leaving as future work?" produced the decision to separate the core from the CLI and defer the service.
- **Gap review:** "What do we still need to define? Are there open contracts?" listed the items that closed the design.
- **Continuous recording:** each approved decision was noted on the whiteboard before moving on, which avoided losing the thread.

## What the AI generated and was corrected

Several AI proposals were changed because I questioned them, and each change improved the design.

| AI proposed | What I decided | Reason |
| --- | --- | --- |
| Contract by function `train()` | Contract by class `MLModel` | Code organization and reuse |
| Schema as a Pydantic attribute in the class | Schema in `schema.yaml`, declared or inferred | Removes the definition from code and allows reviewing the inferred one |
| `.mlt/` folder in the current directory | `.mlt/` anchored to the project | The result depended on where the command was executed |
| Seed only via CLI flag | Flag and environment variable, with recorded origin | Covers scripts and CI, without losing traceability |
| Reproducibility presented as a guarantee | Split into identify, verify, and freeze | Without freezing there is only identity traceability |
| "Predictions the same size as inputs" | Distinction between the number of items and the shape of each prediction | The phrasing was imprecise and masked outputs of another dimension |
| Data statistics in the MVP | Note outside the MVP | They identify the change, but do not reproduce the content |
| Run without code and params.yaml (inconsistency between design and documents) | Self-contained run, with code/ and params.yaml; verify retrains from it | Found in the final review; without this the run does not reproduce if the root changes |
| Full scope for 7 days | Priorities P0, P1, and P2, cutting P2 first | The problem statement does not ask for a complete platform and evaluates prioritization |
| Venv using the "host Python" | Python that runs mlt (sys.executable), without searching for other versions | I asked how this was guaranteed; the term was ambiguous and the result would depend on PATH |
| Python version only via --python flag | Optional .python-version file, with precedence flag, file, default | requirements.txt does not declare Python; the project needs to carry that information |
| check requiring the data/ folder | check consults DataSource; training materializes data in a local directory | I asked whether the snapshot needs to be local; with a lakehouse it would not |
| verify complete (adjustable tolerance, artifact hash) | Minimal version, cuttable; the same argument lives in a test | I questioned the real usefulness; the value is in demonstrating reproducibility |
| serve --detach as P0 | serve --detach as P2, together with ps, stop, and logs | When closing the P2 flows, the dependency appeared: without stop there is no way to end the service |

I also corrected AI layout errors on the whiteboards (text overflowing sections) and asked for a split into two whiteboards when the first became too large.

**Layered contract:** the AI suggested a default `predict` in the core, based on scikit-learn Pipeline, to reduce friction. I questioned the coupling to one artifact type and the difficulty of supporting others, such as PySpark. We decided on an agnostic core with `SklearnModel` as an optional convenience.

## How I verified what the AI produced

In the design, no proposal became a decision without being questioned (see the table above).

In the implementation, verification is done with the critical flow tests and by running the end-to-end example.

### Step 1 — Skeleton, MLModel, contract and CLI (2026-11-24)

**What the AI generated:** pyproject.toml, src/mlt/{core/errors.py, core/contract.py, model/base.py, model/sklearn.py, cli/main.py} and the four test files (37 tests).

**Adjustment made during implementation:** the test `test_no_args_shows_help` failed with exit code 2 because Click returns 2 for a group without a subcommand. The AI fixed it by adding `invoke_without_command=True` and a callback that displays the help; no contract change.

**How the author verified:**
- Ran `pytest` until 37/37 passed.
- Executed `mlt --version` manually and checked the three fields (version, python, docker).
- Read the generated files and confirmed that `MLModel` has no external imports and that joblib/pandas imports in `SklearnModel` are lazy.

**Time spent:** ~25 min (generation + 1 round of test fix).

### Step 2 — venv training, DataSource, RunStore, fingerprint, seed (2026-11-24)

**What the AI generated:** seed.py, fingerprint.py, datasources/{base,local}.py, runstore/{manifest,store}.py, runners/{base,venv}.py, _worker/train.py, core/trainer.py, templates/iris/, conftest.py, and 5 test files (57 new tests, 10 integration).

**No adjustment was necessary:** all 94 tests passed on the first full run.

**How the author verified:**
- Ran `pytest` until 94/94 passed (47 unit + 10 integration + 37 from step 1).
- Inspected the JSON manifest generated in `.mlt/runs/<run_id>/manifest.json` after a real training run with the iris project.
- Read the run's `train.log` to confirm that the worker's stdout/stderr arrive correctly.
- Reviewed `_worker/train.py` and `trainer.py` to confirm that the run is deleted (not marked failed) when the worker exits with code 3.

**Time spent:** ~15 min of generation + 8:36 of integration test execution (first time, downloading packages for the venv).

### Step 3 — Full CLI: init, check, train, runs list/show, diff, verify (2026-11-24)

**What the AI generated:** discovery.py, init_cmd.py, check_cmd.py, train_cmd.py, runs_cmd.py, diff_cmd.py, verify_cmd.py, six test files (42 new tests), and the modifications to main.py, trainer.py, and manifest.py.

**Adjustments made during implementation:**

| What the AI did | What was adjusted | Reason |
| --- | --- | --- |
| Error handling in `main()` only | Moved to `_MltGroup.invoke` | With CliRunner invoking `cli` directly, `MltError` was not caught and exit_code came as 1 instead of the correct code |
| `test_diff_accepts_prefix` using `run_id[:8]` as prefix | Replaced by `test_diff_latest_alias` | The first 8 chars of run_id are the date (`20261124`), identical for all runs on the same day; the test failed due to an ambiguous prefix |
| `test_runs_list_no_runs` without cd to tmp_path | Fixed to do `os.chdir(tmp_path)` | Without this the test depended on the cwd having no runs, which could fail in parallel with other tests |
| `CliRunner(mix_stderr=False)` | Removed the argument | The installed version of Click does not support `mix_stderr` in the constructor |

**No contract or design change was necessary.**

**How the author verified:**
- Ran `pytest` until 136/136 passed (84 previous + 42 new + 10 original integration tests).
- Executed the end-to-end flow manually: `mlt init /tmp/demo --template iris`, `mlt check /tmp/demo`, `mlt train /tmp/demo`, `mlt runs list`, `mlt runs show latest`, `mlt diff <id1> <id2>`, `mlt verify <id>`.
- Confirmed that `mlt runs list` displays `[V]` on the verification run and that `verification_of` is filled in the manifest.

**Time spent:** ~25 min of generation + 11:42 of test execution (136 total; integration tests reused the venv cache).

### Post-step 3 improvement — iris.csv in the template (2026-11-24)

**What the AI generated:** the CSV (via `sklearn.datasets.load_iris`) and the changes to `conftest.py` and `test_cli_init.py`.

**No adjustment was necessary.** `init_cmd.py` already copied everything via `rglob("*")`, so the data was included without code changes.

**How the author verified:** `mlt init /tmp/demo --template iris && mlt train /tmp/demo` complete without a manual data-generation step.

### Test suite overhaul (2026-11-25)

**What the AI generated:** `_force_venv` autouse fixture, `pytest.mark.slow` marker across 5 files, `test_auto_runner_falls_back_to_venv_when_docker_unavailable`, bug fix in `VenvRunner._venv_ready` and cleanup, unified requirements across all slow tests, lazy imports in iris template.

**Adjustments made during implementation:**

| What the AI did | What was adjusted | Reason |
| --- | --- | --- |
| Placed `test_auto_runner_falls_back_to_venv` in `test_trainer.py` (marked slow) | Moved to `test_contract.py` | The test does not build a venv; it should run in the fast profile |
| `_venv_ready` checked only `bin/python` | Also checks `bin/pip` | A broken cached venv (python without pip) caused FileNotFoundError on all subsequent runs |
| `venv_dir.rmdir()` on pip install failure | Changed to `shutil.rmtree` | The venv directory is not empty; `rmdir` silently failed, leaving the broken venv in place |

**How the author verified:**
- Ran `pytest -m 'not docker'`: 143/143 passed in 7:47 (down from 13:08).
- Confirmed three test profiles: `not slow and not docker` (104 tests, seconds), `slow and not docker` (39), `docker` (9).
- Verified that all slow tests share a single venv hash (iris requirements = minimal requirements).
- Confirmed Docker tests share a single image after unifying requirements in `test_docker_runner_deep_check_fails_bad_model`.

### Step 4 — Docker runner and check --deep (2026-11-25)

**What the AI generated:** `DockerRunner` (inline Dockerfile, mlt source in build context, `--network none`), `docker_check.py`, `check --deep` with `--runner [auto|venv|docker]`, `--validate-only` worker flag, `test_runner_docker.py` (9 tests).

**No adjustment record available:** this step was implemented in a prior session without a correction log. Verified by: `pytest -m docker` passing 8/9 tests; the remaining test (`test_docker_runner_deep_check_passes`) had a SIGILL on cold start, fixed in the test suite overhaul by moving sklearn/pandas imports inside `fit()`.

---

### Step 5 — Serving: FastAPI server and mlt serve (2026-11-25)

**What the AI generated:** `src/mlt/serving/{__init__,app,server}.py`, `src/mlt/cli/serve_cmd.py`, `src/mlt/runners/_mlt_pkg.py`, updates to `main.py`, `pyproject.toml` (fastapi, uvicorn, httpx2), and `tests/test_serving.py` (21 tests).

**Corrections and bugs found during author testing:**

| What happened | Fix applied | Reason |
| --- | --- | --- |
| `mlt init .` blocked by `.mypy_cache`, `.python-version` etc. | Replaced named whitelist `{".venv", ".git"}` with rule `name.startswith(".")` | The whitelist only covered two specific names; any other dot-entry blocked init |
| `mlt train` printed `runner  auto` without resolving | Added `auto → docker` / `auto → venv` display using `docker_available()` | "auto" gives no information about what actually ran |
| `mlt train . --template iris` appeared to bring only `model.py` and `.gitignore` | Confirmed all 6 template files are copied; added `[tool.setuptools.package-data]` to `pyproject.toml` | Non-editable wheel installs would have omitted non-`.py` template files |
| `POST /predict` returned 500 for valid iris features in different column order | Added `feature_names_in_` reordering in `SklearnModel.predict` | sklearn >= 1.0 requires feature columns in the same order as training; arbitrary dict key order breaks prediction |
| `mlt train` / `mlt serve` with non-editable mlt install tried to copy all of `site-packages/` to Docker build context | Extracted `_mlt_pkg.py` with `find_mlt_source_root()` and `write_mlt_installable()`; used in both `DockerRunner` and `VenvRunner` | The fallback path in `_find_mlt_root()` returned `.parent.parent` of the package = `site-packages/`; rebuilding after reinstalling mlt required `--force` to pick up the new venv |
| Swagger `/docs` did not show request body editor for `/predict` | Added `openapi_extra` with request body schema and example | Endpoint uses raw `Request`; FastAPI cannot infer body schema automatically |

**How the author verified:**
- Ran `pytest -m "not slow and not docker"`: 126/126 passed.
- Executed the full end-to-end flow manually: `mlt init . --template iris`, `mlt train . --runner venv`, `mlt serve latest`, `curl POST /predict`, `/health`, `/schema`, `/metadata`, Swagger UI at `/docs`.
- Confirmed that `mlt serve` prints `auto → docker` or `auto → venv` in the preamble.
- Confirmed that `POST /predict` with features in any column order returns a valid prediction.

**Time spent:** ~1h (generation + 6 rounds of testing and correction).

---

### Step 6 — Failure modes: timeout, Ctrl+C, build failure, fit exception (2026-11-25)

**What the AI generated:** `except KeyboardInterrupt: ctx.exit(130)` in `_MltGroup.invoke`, and `tests/test_failures.py` (4 fast + 3 slow tests).

**Corrections made during implementation:**

| What happened | Fix applied | Reason |
| --- | --- | --- |
| Fast tests monkeypatched `mlt.cli.train_cmd.train` | Changed target to `mlt.core.trainer.train` | `train_cmd` imports `train` lazily (inside the function body), so the attribute does not exist at module level; the source module must be patched instead |
| Fast tests called CLI with bare `tmp_path` | Added `_stub_project()` that writes `model.py`, `requirements.txt`, `data/` | `require_project_dir()` aborts with exit 2 if `model.py` is absent, before `train()` is ever reached |

**How the author verified:**
- Ran `pytest -m "not slow and not docker"`: 130/130 passed (up from 126).
- Confirmed exit codes via `result.exit_code` in the four fast tests.

**Time spent:** ~15 min (generation + 2 rounds of fix).

---

## Reflections

*(Author's section — fill in before the presentation.)*

- **Where AI helped most:** in building the codebase, once all planning and decisions had already been structured. At that point the AI flows more organically — taking over the writing of code, running tests, iterating, and correcting.
- **Where it required more care:** in the ideation phase, when raising alternatives and building the implementation plan. At that stage there is a more constant exchange between what the AI suggests and what I bring from domain experience — proposing directions that the AI did not originally identify as stronger compared to its own proposals.
- **What surprised me:** the AI was more useful as a design partner (raising what could go wrong, naming the trade-offs) than as a code generator. The code it produced was correct on the first try more often than not, but it required the author's judgment to decide what to build in the first place.
