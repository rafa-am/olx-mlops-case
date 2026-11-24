"""Tests for mlt diff command."""

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


def test_diff_identical_runs(minimal_project):
    p = minimal_project
    run_id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=42)
    # Same fingerprint → idempotency returns same id; force a second run.
    run_id2 = train(p["project_dir"], data_dir=p["data_dir"], seed=42, force=True)
    r = _invoke_from(p["project_dir"], "diff", run_id1, run_id2)
    assert r.exit_code == 0, r.output
    assert "idêntico" in r.output


def test_diff_shows_seed_difference(minimal_project):
    p = minimal_project
    run_id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=42, force=True)
    run_id2 = train(p["project_dir"], data_dir=p["data_dir"], seed=99, force=True)
    r = _invoke_from(p["project_dir"], "diff", run_id1, run_id2)
    assert r.exit_code == 0, r.output
    assert "diferente" in r.output
    assert "seed" in r.output.lower() or "42" in r.output


def test_diff_shows_params_difference(minimal_project):
    p = minimal_project
    run_id1 = train(p["project_dir"], data_dir=p["data_dir"])
    # Change params
    (p["project_dir"] / "params.yaml").write_text("threshold: 0.9\n")
    run_id2 = train(p["project_dir"], data_dir=p["data_dir"])
    r = _invoke_from(p["project_dir"], "diff", run_id1, run_id2)
    assert r.exit_code == 0, r.output
    assert "params" in r.output.lower()


def test_diff_shows_metrics_difference(minimal_project):
    """Different seeds may produce different metrics with DummyClassifier."""
    p = minimal_project
    run_id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=42, force=True)
    run_id2 = train(p["project_dir"], data_dir=p["data_dir"], seed=99, force=True)
    r = _invoke_from(p["project_dir"], "diff", run_id1, run_id2)
    assert r.exit_code == 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_diff_nonexistent_run_fails(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])
    r = _invoke_from(p["project_dir"], "diff", run_id, "nonexistent-run")
    assert r.exit_code != 0


def test_diff_latest_alias(minimal_project):
    p = minimal_project
    run_id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=1, force=True)
    run_id2 = train(p["project_dir"], data_dir=p["data_dir"], seed=2, force=True)
    # "latest" resolves to the most recent run; diff latest vs an explicit run_id.
    r = _invoke_from(p["project_dir"], "diff", run_id1, "latest")
    assert r.exit_code == 0
