"""Tests for flock-based run reconciliation.

Scenarios:
1. Run with no train.lock → not touched (pre-feature run, unknown state).
2. Run with train.lock held by another fd → not orphaned (process alive).
3. Run with train.lock present but unlocked → orphaned, marked failed.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest

from mlt.core.reconcile import is_run_orphaned, reconcile_running_runs
from mlt.runstore.manifest import Manifest
from mlt.runstore.store import RunStore


def _make_running_run(project_dir: Path) -> tuple[RunStore, Manifest]:
    """Create a minimal run with status='running' in the store."""
    store = RunStore(project_dir)
    m = Manifest.new(
        run_id="20991231-000000-test01",
        inputs={"seed": 42, "params": {}, "code": {}, "data": {}, "deps": {}},
        fingerprint="abc123",
        environment={"runner": "venv"},
    )
    store.create_run(m)
    return store, m


# ------------------------------------------------------------------
# is_run_orphaned
# ------------------------------------------------------------------

def test_no_lock_file_is_not_orphaned(tmp_path):
    """A run without train.lock cannot be classified — not orphaned."""
    store, m = _make_running_run(tmp_path)
    run_dir = store.run_dir(m.run_id)
    assert not (run_dir / "train.lock").exists()
    assert is_run_orphaned(run_dir) is False


def test_locked_run_is_not_orphaned(tmp_path):
    """A run whose train.lock is held by another fd is alive — not orphaned."""
    store, m = _make_running_run(tmp_path)
    run_dir = store.run_dir(m.run_id)
    lock_path = run_dir / "train.lock"
    lock_path.touch()

    # Hold an exclusive lock (simulates the training process).
    holder_fd = os.open(str(lock_path), os.O_RDWR)
    fcntl.flock(holder_fd, fcntl.LOCK_EX)
    try:
        assert is_run_orphaned(run_dir) is False
    finally:
        os.close(holder_fd)


def test_unlocked_lock_file_is_orphaned(tmp_path):
    """A run whose train.lock exists but is not held — process died."""
    store, m = _make_running_run(tmp_path)
    run_dir = store.run_dir(m.run_id)
    (run_dir / "train.lock").touch()   # file exists, no one holds it

    assert is_run_orphaned(run_dir) is True


# ------------------------------------------------------------------
# reconcile_running_runs
# ------------------------------------------------------------------

def test_reconcile_fixes_orphaned_run(tmp_path):
    """reconcile_running_runs marks an orphaned run as failed."""
    store, m = _make_running_run(tmp_path)
    run_dir = store.run_dir(m.run_id)
    (run_dir / "train.lock").touch()

    fixed = reconcile_running_runs(tmp_path)

    assert m.run_id in fixed
    updated = store.get_run(m.run_id)
    assert updated.status == "failed"
    assert "interrupted" in (updated.error or "")


def test_reconcile_skips_live_run(tmp_path):
    """reconcile_running_runs leaves a run alone when its lock is held."""
    store, m = _make_running_run(tmp_path)
    run_dir = store.run_dir(m.run_id)
    lock_path = run_dir / "train.lock"
    lock_path.touch()

    holder_fd = os.open(str(lock_path), os.O_RDWR)
    fcntl.flock(holder_fd, fcntl.LOCK_EX)
    try:
        fixed = reconcile_running_runs(tmp_path)
        assert m.run_id not in fixed
        assert store.get_run(m.run_id).status == "running"
    finally:
        os.close(holder_fd)


def test_reconcile_skips_no_lock_file(tmp_path):
    """reconcile_running_runs does not touch runs without train.lock."""
    store, m = _make_running_run(tmp_path)
    # No train.lock — pre-feature run.

    fixed = reconcile_running_runs(tmp_path)

    assert m.run_id not in fixed
    assert store.get_run(m.run_id).status == "running"


def test_reconcile_ignores_terminal_runs(tmp_path):
    """reconcile_running_runs only inspects status=running runs."""
    store = RunStore(tmp_path)
    m = Manifest.new(
        run_id="20991231-000000-test02",
        inputs={"seed": 42, "params": {}, "code": {}, "data": {}, "deps": {}},
        fingerprint="def456",
        environment={"runner": "venv"},
    )
    store.create_run(m)
    store.fail_run(m.run_id, "deliberate failure")
    # Add lock file — should be ignored because status is already 'failed'.
    (store.run_dir(m.run_id) / "train.lock").touch()

    fixed = reconcile_running_runs(tmp_path)
    assert m.run_id not in fixed
