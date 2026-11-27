"""Integration tests for the training orchestrator.

These tests create a real venv, install packages, and run actual training.
The venv is cached at ~/.mlt/envs/<hash>/ so the first run is slow
(package downloads); subsequent runs reuse the cache.
"""

from __future__ import annotations

import shutil

import pytest

from mlt.core.errors import ContractError
from mlt.core.trainer import train
from mlt.runstore.store import RunStore

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_train_succeeds_with_minimal_project(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])
    assert run_id is not None

    store = RunStore(p["project_dir"])
    manifest = store.get_run(run_id)
    assert manifest.status == "succeeded"
    assert manifest.metrics is not None
    assert "accuracy" in manifest.metrics
    assert 0.0 <= manifest.metrics["accuracy"] <= 1.0

    run_dir = store.run_dir(run_id)
    assert (run_dir / "model" / "model.joblib").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "train.log").exists()
    assert (run_dir / "code" / "model.py").exists()
    assert (run_dir / "manifest.json").exists()


def test_train_succeeds_with_iris_project(iris_project):
    p = iris_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])

    store = RunStore(p["project_dir"])
    m = store.get_run(run_id)
    assert m.status == "succeeded"
    assert m.metrics["accuracy"] > 0.5  # sanity check


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_train_idempotent(minimal_project):
    """Same inputs → same run_id returned without re-training."""
    p = minimal_project
    run_id1 = train(p["project_dir"], data_dir=p["data_dir"])
    run_id2 = train(p["project_dir"], data_dir=p["data_dir"])
    assert run_id1 == run_id2


def test_train_force_creates_new_run(minimal_project):
    p = minimal_project
    run_id1 = train(p["project_dir"], data_dir=p["data_dir"])
    run_id2 = train(p["project_dir"], data_dir=p["data_dir"], force=True)
    assert run_id1 != run_id2

    store = RunStore(p["project_dir"])
    assert store.get_run(run_id1).status == "succeeded"
    assert store.get_run(run_id2).status == "succeeded"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_train_same_seed_same_fingerprint(minimal_project):
    p = minimal_project
    run_id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=42, force=True)
    run_id2 = train(p["project_dir"], data_dir=p["data_dir"], seed=42, force=True)

    store = RunStore(p["project_dir"])
    fp1 = store.get_run(run_id1).fingerprint
    fp2 = store.get_run(run_id2).fingerprint
    assert fp1 == fp2


def test_train_different_seed_different_fingerprint(minimal_project):
    p = minimal_project
    run_id1 = train(p["project_dir"], data_dir=p["data_dir"], seed=42, force=True)
    run_id2 = train(p["project_dir"], data_dir=p["data_dir"], seed=99, force=True)

    store = RunStore(p["project_dir"])
    assert store.get_run(run_id1).fingerprint != store.get_run(run_id2).fingerprint


# ---------------------------------------------------------------------------
# Params change → new fingerprint
# ---------------------------------------------------------------------------


def test_train_params_change_new_run(minimal_project):
    p = minimal_project
    run_id1 = train(p["project_dir"], data_dir=p["data_dir"])

    # Change params.yaml
    (p["project_dir"] / "params.yaml").write_text("threshold: 0.5\n")
    run_id2 = train(p["project_dir"], data_dir=p["data_dir"])

    assert run_id1 != run_id2
    store = RunStore(p["project_dir"])
    assert store.get_run(run_id1).fingerprint != store.get_run(run_id2).fingerprint


# ---------------------------------------------------------------------------
# Contract errors
# ---------------------------------------------------------------------------


def test_train_contract_error_no_succeeded_run(tmp_path):
    """Static contract violation → ContractError, no run created."""
    # model.py with no MLModel subclass
    (tmp_path / "model.py").write_text("class NotAModel: pass\n")
    (tmp_path / "requirements.txt").write_text("scikit-learn>=1.3\n")
    data = tmp_path / "data"
    data.mkdir()
    (data / "x.csv").write_text("x\n1\n")

    with pytest.raises(ContractError):
        train(tmp_path, data_dir=data)

    store = RunStore(tmp_path)
    succeeded = store.list_runs()
    assert all(m.status != "succeeded" for m in succeeded)


def test_train_missing_data_raises_contract_error(minimal_project):
    p = minimal_project
    with pytest.raises(ContractError, match="not found"):
        train(p["project_dir"], data_dir=p["project_dir"] / "nonexistent")


# ---------------------------------------------------------------------------
# Manifest completeness
# ---------------------------------------------------------------------------


def test_manifest_has_all_required_fields(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])

    store = RunStore(p["project_dir"])
    m = store.get_run(run_id)
    d = m.to_dict()

    assert d["manifest_version"] == 1
    assert d["run_id"] == run_id
    assert d["status"] == "succeeded"
    assert d["fingerprint"] is not None
    assert d["inputs"]["seed"] == 42  # default
    assert d["inputs"]["deps"]["lock"]  # pip freeze not empty
    assert d["environment"]["runner"] == "venv"
    assert d["outputs"]["metrics"]


# ---------------------------------------------------------------------------
# Build log
# ---------------------------------------------------------------------------


def test_build_log_archived_in_run_dir(minimal_project):
    """When the venv is built fresh, build.log ends up inside run_dir."""
    from mlt.runners.venv import VenvRunner, _ENVS_DIR

    p = minimal_project

    # Evict the cached env so a real build is forced.
    runner = VenvRunner(p["project_dir"] / "requirements.txt", "3.10")
    env_dir = _ENVS_DIR / runner.get_env_hash()
    if env_dir.exists():
        shutil.rmtree(env_dir)

    run_id = train(p["project_dir"], data_dir=p["data_dir"], force=True)
    store = RunStore(p["project_dir"])
    run_dir = store.run_dir(run_id)

    assert (run_dir / "build.log").exists(), "build.log should be archived in run_dir"
    assert (run_dir / "build.log").stat().st_size > 0, "build.log should be non-empty"
    # Verify it is gone from .mlt/ (moved, not copied).
    assert not (p["project_dir"] / ".mlt" / "build.log").exists()


def test_build_log_absent_on_cache_hit(minimal_project):
    """When the venv is already cached, build.log is not created in run_dir."""
    p = minimal_project

    # First run warms the cache.
    train(p["project_dir"], data_dir=p["data_dir"])

    # Second run (force=True, different seed → new run_id) reuses cached env.
    run_id = train(p["project_dir"], data_dir=p["data_dir"], seed=99, force=True)
    store = RunStore(p["project_dir"])
    run_dir = store.run_dir(run_id)

    # build.log must NOT appear — no build happened.
    assert not (run_dir / "build.log").exists()
