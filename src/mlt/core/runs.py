"""Core logic for listing and inspecting runs.

Returns structured data (Manifest objects); never prints.
Raises mlt.core.errors typed exceptions on bad input.
"""

from __future__ import annotations

from pathlib import Path

from mlt.runstore.manifest import Manifest
from mlt.runstore.store import RunStore


def list_runs(
    project_dir: Path,
    limit: int = 20,
    show_all: bool = False,
) -> list[Manifest]:
    """Return runs for *project_dir*, newest first."""
    store = RunStore(project_dir)
    return store.list_runs(limit=limit, show_all=show_all)


def describe_run(project_dir: Path, run_id: str) -> Manifest:
    """Return the manifest for *run_id* (accepts prefix and 'latest').

    Raises NotFoundError (exit 7) if the run does not exist or the
    prefix is ambiguous.
    """
    store = RunStore(project_dir)
    return store.get_run(run_id)
