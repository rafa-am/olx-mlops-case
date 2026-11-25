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

## Reflections

Author's draft, to be revised at the end of the implementation.

- **Where AI helped most:** raising alternatives and trade-offs, and listing what was still open.
- **Where it required more care:** guarantee statements (reproducibility, traceability) and format details, which I had to question to get them precise.
