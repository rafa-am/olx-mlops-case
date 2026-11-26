# CLAUDE.md

Project context for Claude Code.

## What it is

`mlt`: a Python CLI package that takes a data scientist from reproducible, isolated training to a REST inference API, from Python code alone. This is a technical case study for a Machine Learning Engineer position. The focus is the tool, not the model.

Reference documents, in order of authority:

1. `docs/design.md`: what to build (contracts, manifest, API, exit codes, build order).
2. `DECISIONS.md`: why each decision was made and the alternatives discarded.
3. `AI_USAGE.md`: AI usage log (see rule below).

If anything in the code contradicts `docs/design.md`, stop and ask before proceeding. If a decision changes during implementation, update `docs/design.md` and `DECISIONS.md` in the same commit.

## Working rules

- Work one step at a time, in the order of `docs/design.md` section 16. Plan before each step and wait for plan review.
- Respect the P0/P1/P2 priorities from section 16: do P0 first; P2 only if time permits. Record time spent per step and what was cut.
- The run stores `code/` (copy of the project without `data/` and `.mlt/`) and `params.yaml`; `verify` retrains from `code/`. `mlt` enters the Docker image copied to the build context (hash includes `mlt` source code).
- Small commits, one per step or behavior, with tests passing before each. Commit messages in English, imperative mood, short.
- Write tests for critical flows (section 15) alongside each step, not after.
- Do not add dependencies or features outside scope (section 17) without asking.
- Simple and readable code: the author will defend each decision for 30 minutes. Explain non-obvious choices in a short comment.
- No secrets in the repository. `.mlt/`, `~/.mlt`, caches and generated data go in `.gitignore`.

## Design rules that cannot be violated

- The CLI is a thin layer only: logic lives in `mlt.core`. Exit codes are an enumeration from the core.
- The `src/mlt/model/` core (`MLModel`, with abstract `fit`, `predict`, `save` and `load`) imports nothing outside the standard library and brings no framework-tied patterns. Conveniences like `SklearnModel` (joblib and pandas) live in their own modules, with lazy imports. The iris template inherits from `SklearnModel`.
- Training runs without network and with read-only data. Data never enters the Docker image.
- The run is immutable. The manifest is the source of truth; SQLite is just a rebuildable index.
- The fingerprint is computed only over `inputs`. The code hash ignores `.mlt/`, `.git/`, `__pycache__/` and data.
- Root is input, run is record: `serve`, `verify` and `diff` read from the run; `check` reads from the root.
- Serving always obtains the run via `RunStore`, never reading `runs/` directly.
- `check` is static (does not import `model.py` on the host). Deep validation runs inside the environment, before creating the run.
- `save` and `load` operate on a local directory; transport to a remote registry is the `RunStore`'s responsibility, never the model's.
- Contract error (code 3) does not create a run. Execution error (4) and timeout (5) create a `failed` run.
- `.mlt/` lives inside the scientist's project, never in the current directory by accident.
- The README includes the scientist's guide (one page, see `docs/design.md` section 4): which base class to choose, what `fit` and `predict` receive, complete `requirements.txt` and data in `<project>/data/` or `--data`. The `iris` and `blank` templates have short comments on each method.
- Do not promise what cannot be guaranteed: without `--snapshot-data`, data traceability is identity-level, not content-level.

## Structure

```
src/mlt/{cli,core,model,runners,datasources,runstore,serving,templates/{iris,blank}}
tests/
docs/{design.md,architecture.md}
README.md  DECISIONS.md  AI_USAGE.md
```

## Commands

- Install for development: `pip install -e ".[dev]"`
- Tests: `pytest` (Docker tests are marked `docker` and skipped without Docker; `pytest -m docker` runs them only)
- End-to-end example: `mlt init /tmp/demo --template iris && mlt train /tmp/demo && mlt runs list`
- Formatting and lint: to be defined in step 1 (suggestion: ruff).

## AI usage log (required)

At the end of each step, add to `AI_USAGE.md` (section "What the AI generated and was corrected") any point where the author corrected, rejected, or adjusted what was generated, with the reason. If no correction was made, state that in the response so the author can record what was verified and how.

## Out of scope

Authentication, data versioning, rollback and A/B, monitoring, GPU, distributed training, UI, CPU and memory limits, run cleanup, route versioning, data statistics in the manifest. The cloud (AWS) is design only.

- Installation: Python `>=3.10` on the host (`pyproject.toml`). `mlt --version` shows version, Python and Docker detection (available and daemon active). The README lists Docker running (optional) and network on first build as prerequisites.
- `mlt init <folder>`: creates a new folder, refuses non-empty folder (exit 2), generates `.gitignore` with `.mlt/`, does not run `git init`, prints next steps.
- `check`: data via `DataSource.check()` (MVP: local folder exists and non-empty; error exit 3); indirect inheritance produces a warning, not an error; output `ok`/`warning`/`error` with file and line. Project discovery walks up to `model.py` or `.mlt/`; `runs list` with no runs responds exit 0.
- `train`: final summary with run_id, status, metrics and next action; slow build warning; identical run does not train (exit 0, `--force` repeats); Ctrl+C marks run `failed` (`interrupted`) and exits 130; build failure exits 6 without creating a run.
- Python: precedence `--python` > `.python-version` > default (Docker 3.12; venv uses `sys.executable`, without searching PATH; different version in venv exits 6). Major.minor enters the `deps` fingerprint. Runner is outside the fingerprint (informational only). `train` and `check` print effective values and their origin (python, runner, seed). Without Docker: `auto` falls back to venv with a warning, explicit `docker` exits 6.
- `runs list`: one line per run (run_id, status, when, duration, up to 3 metrics, short fingerprint), limit 20, `--all`. `runs show`: sections identity/inputs/environment/outputs/error; `--json` = manifest. Every command with `run_id` accepts `latest` and unique prefix; non-existent or ambiguous exits 7.
- `serve`: prints URL, run_id, /docs and filled `curl` only after model loads and /health returns 200; default port 8000, occupied exits 6 (suggests --port, does not auto-switch); /docs kept; schema read from run, never from root.
- `diff <a> <b>`: reads manifests, starts with 'fingerprint: different (components)', shows only what differs. `verify <run_id>`: minimal version; retrains only from what the run stored (`code/`, params, schema, seed, Python) with the same runner; creates run with `verification_of`; compares fingerprint and metrics (fixed relative tolerance 1e-6); exit 0 reproduced, 4 diverged, 8 if data changed without snapshot. No `--tolerance` or artifact hash. Cuttable: the integration test 'trains twice' covers the argument.
- P2 with closed flow in `docs/design.md` (before section 17): `--snapshot-data` (content-addressed store, `frozen`), `serve --detach` + `ps`/`stop`/`logs` (PID in `.mlt/serving/`), reconciliation via `flock`, `blank` template (marker `# TODO(mlt)`), `--json` (list, show, diff, verify), `runs list --status`. `init --force` will not be done. `serve --detach` is P2: P0 is foreground only.
