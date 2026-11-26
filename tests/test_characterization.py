"""Characterization tests for runs list/show, diff and verify.

These tests fix the structural invariants of the CLI output — fields that
must be present, exit codes — without pinning exact text byte-for-byte.
They act as a safety net for the Fase 1 core extraction: each commit that
moves logic from a cmd module to mlt.core must leave these tests green.

All tests are marked slow because they run real training via the venv runner.
"""

from __future__ import annotations

import os
import textwrap

import pytest
from click.testing import CliRunner

from mlt.cli.main import cli
from mlt.core.trainer import train
from mlt.runstore.store import RunStore

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Shared session fixture — trains once, reused by all tests in this file
# ---------------------------------------------------------------------------

_MINIMAL_MODEL = """\
    from mlt.model.sklearn import SklearnModel

    class Model(SklearnModel):
        def fit(self, data_dir, params):
            from sklearn.dummy import DummyClassifier
            import pandas as pd
            df = pd.read_csv(data_dir / "data.csv")
            X = df[["x"]]
            y = df["y"]
            self.model = DummyClassifier(strategy="most_frequent")
            self.model.fit(X, y)
            return {"accuracy": float(self.model.score(X, y))}
"""

_MINIMAL_REQ = "scikit-learn>=1.3\npandas>=2.0\nnumpy>=1.24\n"


@pytest.fixture(scope="session")
def char_project(tmp_path_factory):
    """Session-scoped project with three runs:

    - run_a: succeeded, seed=42
    - run_b: succeeded, seed=99 (different seed → different fingerprint)
    - run_failed: failed (forced via store.fail_run)

    The project lives in a directory outside any git repo, so git_commit
    is None in every run (the "no-git" case).
    """
    project_dir = tmp_path_factory.mktemp("char_proj")
    (project_dir / "model.py").write_text(textwrap.dedent(_MINIMAL_MODEL))
    (project_dir / "requirements.txt").write_text(_MINIMAL_REQ)
    data_dir = project_dir / "data"
    data_dir.mkdir()
    csv = "x,y\n" + "\n".join(f"{i},{i % 2}" for i in range(10)) + "\n"
    (data_dir / "data.csv").write_text(csv)

    # Use venv explicitly so verify tests work without Docker.
    run_a = train(project_dir, data_dir=data_dir, seed=42, runner="venv")
    run_b = train(project_dir, data_dir=data_dir, seed=99, force=True, runner="venv")

    # Create a failed run by training first then forcing failure via the store.
    run_c = train(project_dir, data_dir=data_dir, seed=7, force=True, runner="venv")
    RunStore(project_dir).fail_run(run_c, "forced failure for characterization test")

    return {
        "project_dir": project_dir,
        "data_dir": data_dir,
        "run_a": run_a,
        "run_b": run_b,
        "run_failed": run_c,
    }


def _invoke(project_dir, *args):
    """Invoke the CLI from project_dir, return CliRunner result."""
    old = os.getcwd()
    os.chdir(project_dir)
    try:
        return CliRunner().invoke(cli, list(args))
    finally:
        os.chdir(old)


# ---------------------------------------------------------------------------
# runs list
# ---------------------------------------------------------------------------


def test_runs_list_exit_code(char_project):
    r = _invoke(char_project["project_dir"], "runs", "list")
    assert r.exit_code == 0, r.output


def test_runs_list_shows_all_three_runs(char_project):
    r = _invoke(char_project["project_dir"], "runs", "list")
    output = r.output
    # Each run_id prefix must appear
    assert char_project["run_a"][:8] in output
    assert char_project["run_b"][:8] in output
    assert char_project["run_failed"][:8] in output


def test_runs_list_shows_succeeded_and_failed(char_project):
    r = _invoke(char_project["project_dir"], "runs", "list")
    assert "succeeded" in r.output
    assert "failed" in r.output


def test_runs_list_shows_metrics(char_project):
    r = _invoke(char_project["project_dir"], "runs", "list")
    # DummyClassifier returns accuracy; it must appear for succeeded runs
    assert "accuracy" in r.output


# ---------------------------------------------------------------------------
# runs show — succeeded run
# ---------------------------------------------------------------------------


def test_runs_show_succeeded_exit_code(char_project):
    r = _invoke(char_project["project_dir"], "runs", "show", char_project["run_a"])
    assert r.exit_code == 0, r.output


def test_runs_show_succeeded_identity_fields(char_project):
    r = _invoke(char_project["project_dir"], "runs", "show", char_project["run_a"])
    assert "run_id" in r.output
    assert "status" in r.output
    assert "succeeded" in r.output


def test_runs_show_succeeded_sections(char_project):
    r = _invoke(char_project["project_dir"], "runs", "show", char_project["run_a"])
    assert "INPUTS" in r.output
    assert "ENVIRONMENT" in r.output
    assert "OUTPUTS" in r.output


def test_runs_show_succeeded_has_metrics(char_project):
    r = _invoke(char_project["project_dir"], "runs", "show", char_project["run_a"])
    assert "accuracy" in r.output


# ---------------------------------------------------------------------------
# runs show — no-git run (project not inside a git repo → commit: -)
# ---------------------------------------------------------------------------


def test_runs_show_no_git_commit_dash(char_project):
    """Because char_project is not inside a git repo, commit must show '-'."""
    r = _invoke(char_project["project_dir"], "runs", "show", char_project["run_a"])
    assert "commit: -" in r.output


# ---------------------------------------------------------------------------
# runs show — failed run
# ---------------------------------------------------------------------------


def test_runs_show_failed_exit_code(char_project):
    r = _invoke(char_project["project_dir"], "runs", "show", char_project["run_failed"])
    assert r.exit_code == 0, r.output


def test_runs_show_failed_has_error_section(char_project):
    r = _invoke(char_project["project_dir"], "runs", "show", char_project["run_failed"])
    assert "ERROR" in r.output


def test_runs_show_failed_contains_error_message(char_project):
    r = _invoke(char_project["project_dir"], "runs", "show", char_project["run_failed"])
    assert "forced failure" in r.output


def test_runs_show_failed_has_status_failed(char_project):
    r = _invoke(char_project["project_dir"], "runs", "show", char_project["run_failed"])
    assert "failed" in r.output


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def test_diff_exit_code(char_project):
    r = _invoke(
        char_project["project_dir"], "diff",
        char_project["run_a"], char_project["run_b"],
    )
    assert r.exit_code == 0, r.output


def test_diff_starts_with_fingerprint_line(char_project):
    r = _invoke(
        char_project["project_dir"], "diff",
        char_project["run_a"], char_project["run_b"],
    )
    assert r.output.startswith("fingerprint:"), r.output


def test_diff_different_runs_shows_different(char_project):
    """run_a and run_b have different seeds → fingerprint must differ."""
    r = _invoke(
        char_project["project_dir"], "diff",
        char_project["run_a"], char_project["run_b"],
    )
    assert "different" in r.output


def test_diff_different_runs_names_seed_component(char_project):
    r = _invoke(
        char_project["project_dir"], "diff",
        char_project["run_a"], char_project["run_b"],
    )
    assert "seed" in r.output


def test_diff_identical_runs_shows_identical(char_project):
    """Same run diffed against itself → identical."""
    r = _invoke(
        char_project["project_dir"], "diff",
        char_project["run_a"], char_project["run_a"],
    )
    assert r.exit_code == 0
    assert "identical" in r.output


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_exit_code_zero(char_project):
    r = _invoke(
        char_project["project_dir"],
        "verify", char_project["run_a"],
        "--data", str(char_project["data_dir"]),
    )
    assert r.exit_code == 0, r.output


def test_verify_reports_reproducible(char_project):
    r = _invoke(
        char_project["project_dir"],
        "verify", char_project["run_a"],
        "--data", str(char_project["data_dir"]),
    )
    assert "reproducible" in r.output


def test_verify_creates_verification_run(char_project):
    """After verify, runs list must show a [V] marker."""
    _invoke(
        char_project["project_dir"],
        "verify", char_project["run_a"],
        "--data", str(char_project["data_dir"]),
    )
    r = _invoke(char_project["project_dir"], "runs", "list")
    assert "[V]" in r.output
