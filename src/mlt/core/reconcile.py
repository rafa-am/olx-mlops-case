"""Reconcile runs stuck in 'running' status after a process crash.

How it works
------------
When a training job starts, trainer.py acquires an exclusive flock on
``<run_dir>/train.lock`` and holds it until the job ends (success, failure,
or any exception including KeyboardInterrupt).

If the process dies abruptly (kill -9, power loss, server crash), the OS
releases all file locks automatically.  Any other process can then acquire
a non-blocking exclusive flock on the same file, which proves the original
process is gone.  ``reconcile_running_runs`` uses this test to find and
fix orphaned runs.

Limitations
-----------
- Only Unix (uses ``fcntl``).
- Runs created before this feature was added have no ``train.lock``;
  they are left untouched (we cannot safely determine their liveness).
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


def is_run_orphaned(run_dir: Path) -> bool:
    """Return True when *run_dir* has a ``train.lock`` whose owner is dead."""
    lock_path = run_dir / "train.lock"
    if not lock_path.exists():
        # No lock file: pre-reconciliation run; we cannot tell.
        return False
    try:
        fd = os.open(str(lock_path), os.O_RDWR)
        try:
            # Non-blocking acquire: succeeds only if no one holds the lock.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)  # release immediately
            return True  # acquired → original process is dead
        except (BlockingIOError, OSError):
            return False  # lock held → process is alive
        finally:
            os.close(fd)
    except OSError:
        return False


def reconcile_running_runs(project_dir: Path) -> list[str]:
    """Mark orphaned 'running' runs as failed; return their run_ids.

    Safe to call at startup or from any read path, as long as the caller
    is not the process currently holding the lock (which would be a bug,
    not a crash).
    """
    from mlt.runstore.store import RunStore

    store = RunStore(project_dir)
    fixed: list[str] = []
    for m in store.list_runs(show_all=True):
        if m.status != "running":
            continue
        run_dir = store.run_dir(m.run_id)
        if is_run_orphaned(run_dir):
            store.fail_run(m.run_id, "interrupted: process died unexpectedly")
            fixed.append(m.run_id)
    return fixed
