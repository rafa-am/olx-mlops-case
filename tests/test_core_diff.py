"""Unit tests for mlt.core.diff — no CliRunner."""

from __future__ import annotations

import pytest

from mlt.core.diff import RunDiff, diff_runs, diff_runs_by_id
from mlt.core.errors import NotFoundError
from mlt.core.trainer import train
from mlt.runstore.manifest import Manifest

pytestmark = pytest.mark.slow


def test_diff_runs_returns_rundiff(minimal_project):
    p = minimal_project
    id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=1, force=True)
    id2 = train(p["project_dir"], data_dir=p["data_dir"], seed=2, force=True)
    store_get = __import__("mlt.runstore.store", fromlist=["RunStore"]).RunStore
    store = store_get(p["project_dir"])
    m1 = store.get_run(id1)
    m2 = store.get_run(id2)
    result = diff_runs(m1, m2)
    assert isinstance(result, RunDiff)


def test_diff_identical_runs_fingerprint_match(minimal_project):
    p = minimal_project
    id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=42, force=True)
    id2 = train(p["project_dir"], data_dir=p["data_dir"], seed=42, force=True)
    from mlt.runstore.store import RunStore
    store = RunStore(p["project_dir"])
    result = diff_runs(store.get_run(id1), store.get_run(id2))
    assert result.fingerprint_match
    assert result.fingerprint_components == []
    assert not result.any_diff


def test_diff_different_seed_detected(minimal_project):
    p = minimal_project
    id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=1, force=True)
    id2 = train(p["project_dir"], data_dir=p["data_dir"], seed=2, force=True)
    from mlt.runstore.store import RunStore
    store = RunStore(p["project_dir"])
    result = diff_runs(store.get_run(id1), store.get_run(id2))
    assert not result.fingerprint_match
    assert "seed" in result.fingerprint_components
    assert result.seed_diff == (1, 2)
    assert result.any_diff


def test_diff_runs_by_id_resolves_ids(minimal_project):
    p = minimal_project
    id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=10, force=True)
    id2 = train(p["project_dir"], data_dir=p["data_dir"], seed=20, force=True)
    result, m1, m2 = diff_runs_by_id(p["project_dir"], id1, id2)
    assert isinstance(result, RunDiff)
    assert isinstance(m1, Manifest)
    assert isinstance(m2, Manifest)
    assert m1.run_id == id1
    assert m2.run_id == id2


def test_diff_runs_by_id_accepts_latest(minimal_project):
    p = minimal_project
    id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=10, force=True)
    train(p["project_dir"], data_dir=p["data_dir"], seed=20, force=True)
    # "latest" resolves to the most recent run.
    result, _, m2 = diff_runs_by_id(p["project_dir"], id1, "latest")
    assert isinstance(result, RunDiff)
    assert m2.run_id != id1


def test_diff_runs_by_id_nonexistent_raises(minimal_project):
    p = minimal_project
    train(p["project_dir"], data_dir=p["data_dir"])
    with pytest.raises(NotFoundError):
        diff_runs_by_id(p["project_dir"], "nonexistent", "also-nonexistent")
