"""Tests for mlt verify command.

The integration test creates a real venv and trains twice; it reuses the
env cache so only the first run is slow.
"""

import os

import pytest
from click.testing import CliRunner

from mlt.cli.main import cli
from mlt.core.trainer import train


def _invoke_from(project_dir, *args):
    old = os.getcwd()
    os.chdir(project_dir)
    try:
        return CliRunner().invoke(cli, list(args))
    finally:
        os.chdir(old)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_verify_reproduces_run(minimal_project):
    """verify should succeed for a run trained with an explicit seed."""
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"], seed=42)
    r = _invoke_from(
        p["project_dir"],
        "verify", run_id, "--data", str(p["data_dir"]),
    )
    assert r.exit_code == 0, r.output
    assert "reproduzível" in r.output


def test_verify_creates_verification_run(minimal_project):
    """The verification run should appear in runs list with [V] marker."""
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"], seed=42)
    _invoke_from(p["project_dir"], "verify", run_id, "--data", str(p["data_dir"]))

    # Check runs list shows a verification run
    r = _invoke_from(p["project_dir"], "runs", "list")
    assert "[V]" in r.output


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_verify_refuses_failed_run(minimal_project):
    """verify on a non-succeeded run should exit non-zero."""
    # We can't easily create a failed run via the CLI, so use the store directly.
    from mlt.runstore.store import RunStore

    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])
    store = RunStore(p["project_dir"])
    store.fail_run(run_id, "forced failure for test")

    r = _invoke_from(
        p["project_dir"],
        "verify", run_id, "--data", str(p["data_dir"]),
    )
    assert r.exit_code != 0
    assert "error:" in r.output or "refused" in r.output.lower()


def test_verify_refuses_changed_data(minimal_project):
    """verify should exit non-zero when the data hash has changed."""
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])

    # Modify the data dir to change its hash.
    (p["data_dir"] / "extra.csv").write_text("x\n1\n2\n")

    r = _invoke_from(
        p["project_dir"],
        "verify", run_id, "--data", str(p["data_dir"]),
    )
    assert r.exit_code != 0
    assert "error:" in r.output or "data" in r.output.lower()


def test_verify_nonexistent_run_fails(minimal_project):
    p = minimal_project
    r = _invoke_from(p["project_dir"], "verify", "nonexistent-run")
    assert r.exit_code != 0
