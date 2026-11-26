"""Unit tests for mlt.core.runs — no CliRunner, no CLI layer."""

from __future__ import annotations

import pytest

from mlt.core.runs import describe_run, list_runs
from mlt.core.errors import NotFoundError
from mlt.core.trainer import train
from mlt.runstore.manifest import Manifest

pytestmark = pytest.mark.slow


def test_list_runs_returns_list_of_manifests(minimal_project):
    p = minimal_project
    train(p["project_dir"], data_dir=p["data_dir"])
    runs = list_runs(p["project_dir"])
    assert isinstance(runs, list)
    assert len(runs) >= 1
    assert all(isinstance(m, Manifest) for m in runs)


def test_list_runs_empty_project_returns_empty(tmp_path):
    # A project with no runs returns an empty list (no exception).
    (tmp_path / "model.py").touch()
    runs = list_runs(tmp_path)
    assert runs == []


def test_list_runs_newest_first(minimal_project):
    p = minimal_project
    train(p["project_dir"], data_dir=p["data_dir"], seed=1, force=True)
    train(p["project_dir"], data_dir=p["data_dir"], seed=2, force=True)
    runs = list_runs(p["project_dir"])
    # created_at timestamps should be non-increasing (newest first)
    dates = [m.created_at for m in runs]
    assert dates == sorted(dates, reverse=True)


def test_describe_run_returns_manifest(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])
    m = describe_run(p["project_dir"], run_id)
    assert isinstance(m, Manifest)
    assert m.run_id == run_id


def test_describe_run_accepts_prefix(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])
    m = describe_run(p["project_dir"], run_id[:8])
    assert m.run_id == run_id


def test_describe_run_accepts_latest(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])
    m = describe_run(p["project_dir"], "latest")
    assert m.run_id == run_id


def test_describe_run_nonexistent_raises_not_found(minimal_project):
    p = minimal_project
    with pytest.raises(NotFoundError):
        describe_run(p["project_dir"], "nonexistent-run-id")
