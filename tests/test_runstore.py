"""Tests for RunStore and Manifest."""

import json
from datetime import datetime, timezone

import pytest

from mlt.runstore.manifest import Manifest
from mlt.runstore.store import RunStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(run_id: str = "20260101-000000-aabbcc") -> Manifest:
    return Manifest.new(
        run_id=run_id,
        inputs={
            "code": {"hash": "ch", "git_commit": None, "dirty": None},
            "params": {"hash": "ph", "values": {}},
            "schema": {"hash": None, "source": "none"},
            "python": {"requested": "3.10", "source": "default"},
            "data": {"hash": "dh", "source": "local", "files": {}, "snapshot": None, "frozen": False},
            "deps": {"hash": "deph", "lock": []},
            "seed": 42,
            "seed_source": "default",
        },
        fingerprint="fp_abc",
        environment={"runner": "venv", "python": "3.10.12", "isolation": "none", "env_hash": "eh"},
    )


# ---------------------------------------------------------------------------
# Manifest serialisation
# ---------------------------------------------------------------------------


def test_manifest_roundtrip(tmp_path):
    m = _make_manifest()
    path = tmp_path / "manifest.json"
    m.write(path)
    m2 = Manifest.read(path)
    assert m2.run_id == m.run_id
    assert m2.status == "running"
    assert m2.fingerprint == "fp_abc"


def test_manifest_atomic_write_uses_tmp(tmp_path):
    """write() must rename from .tmp, not write directly."""
    m = _make_manifest()
    path = tmp_path / "manifest.json"
    m.write(path)
    assert path.exists()
    assert not (tmp_path / "manifest.tmp").exists()


def test_manifest_mark_succeeded():
    m = _make_manifest()
    m.mark_succeeded(
        metrics={"accuracy": 0.97},
        schema_outputs={"hash": "sh", "source": "inferred"},
        artifacts={"model/model.joblib": "arth"},
        deps_lock=["scikit-learn==1.5.2"],
        env_hash="eh",
    )
    assert m.status == "succeeded"
    assert m.metrics == {"accuracy": 0.97}
    assert m.duration_s is not None


def test_manifest_mark_failed():
    m = _make_manifest()
    m.mark_failed("fit raised ValueError")
    assert m.status == "failed"
    assert m.error == "fit raised ValueError"
    assert m.duration_s is not None


# ---------------------------------------------------------------------------
# RunStore CRUD
# ---------------------------------------------------------------------------


def test_runstore_create_run(tmp_path):
    store = RunStore(tmp_path)
    m = _make_manifest()
    run_dir = store.create_run(m)
    assert run_dir.exists()
    assert (run_dir / "manifest.json").exists()


def test_runstore_create_run_mkdir_exclusivity(tmp_path):
    """Creating the same run_id twice must fail."""
    store = RunStore(tmp_path)
    m = _make_manifest()
    store.create_run(m)
    with pytest.raises(FileExistsError):
        store.create_run(m)


def test_runstore_complete_run(tmp_path):
    store = RunStore(tmp_path)
    m = _make_manifest()
    store.create_run(m)
    m.mark_succeeded(
        metrics={"acc": 0.9},
        schema_outputs={"hash": None, "source": "none"},
        artifacts={},
        deps_lock=[],
        env_hash="eh",
    )
    store.complete_run(m)
    loaded = store.get_run(m.run_id)
    assert loaded.status == "succeeded"
    assert loaded.metrics == {"acc": 0.9}


def test_runstore_fail_run(tmp_path):
    store = RunStore(tmp_path)
    m = _make_manifest()
    store.create_run(m)
    store.fail_run(m.run_id, "test error")
    loaded = store.get_run(m.run_id)
    assert loaded.status == "failed"
    assert loaded.error == "test error"


def test_runstore_get_by_prefix(tmp_path):
    store = RunStore(tmp_path)
    m = _make_manifest("20260101-000000-aabbcc")
    store.create_run(m)
    loaded = store.get_run("20260101")  # prefix
    assert loaded.run_id == "20260101-000000-aabbcc"


def test_runstore_get_latest(tmp_path):
    store = RunStore(tmp_path)
    m1 = _make_manifest("20260101-000000-aabbcc")
    m2 = _make_manifest("20260102-000000-ddeeff")
    store.create_run(m1)
    store.create_run(m2)
    loaded = store.get_run("latest")
    assert loaded.run_id == "20260102-000000-ddeeff"


def test_runstore_not_found_raises(tmp_path):
    from mlt.core.errors import NotFoundError

    store = RunStore(tmp_path)
    with pytest.raises(NotFoundError):
        store.get_run("nonexistent")


def test_runstore_ambiguous_prefix_raises(tmp_path):
    from mlt.core.errors import NotFoundError

    store = RunStore(tmp_path)
    store.create_run(_make_manifest("20260101-000000-aabbcc"))
    store.create_run(_make_manifest("20260101-000000-aabbdd"))
    with pytest.raises(NotFoundError, match="ambiguous"):
        store.get_run("20260101-000000-aabb")


def test_runstore_find_by_fingerprint(tmp_path):
    store = RunStore(tmp_path)
    m = _make_manifest()
    store.create_run(m)
    m.mark_succeeded({"acc": 1.0}, {"hash": None, "source": "none"}, {}, [], "eh")
    store.complete_run(m)
    result = store.find_by_fingerprint("fp_abc")
    assert result is not None
    assert result.run_id == m.run_id


def test_runstore_find_by_fingerprint_not_found(tmp_path):
    store = RunStore(tmp_path)
    assert store.find_by_fingerprint("nonexistent") is None


def test_runstore_list_runs(tmp_path):
    store = RunStore(tmp_path)
    store.create_run(_make_manifest("20260101-000000-aabbcc"))
    store.create_run(_make_manifest("20260102-000000-ddeeff"))
    runs = store.list_runs()
    assert len(runs) == 2
    # Newest first
    assert runs[0].run_id == "20260102-000000-ddeeff"


def test_runstore_list_runs_limit(tmp_path):
    store = RunStore(tmp_path)
    for i in range(5):
        store.create_run(_make_manifest(f"2026010{i}-000000-aabbcc"))
    runs = store.list_runs(limit=3)
    assert len(runs) == 3


def test_runstore_remove_run(tmp_path):
    from mlt.core.errors import NotFoundError

    store = RunStore(tmp_path)
    m = _make_manifest()
    store.create_run(m)
    store.remove_run(m.run_id)
    with pytest.raises(NotFoundError):
        store.get_run(m.run_id)
