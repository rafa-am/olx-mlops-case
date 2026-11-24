"""RunStore: file layout + SQLite index.

Layout under ``<project>/.mlt/``:
  runs/<run_id>/manifest.json
  runs/<run_id>/model/
  runs/<run_id>/metrics.json
  runs/<run_id>/params.yaml      (effective)
  runs/<run_id>/schema.yaml      (effective)
  runs/<run_id>/code/            (project snapshot for verify)
  runs/<run_id>/train.log
  index.sqlite                   (query index; manifests are the truth)

The SQLite index is in WAL mode so concurrent readers don't block writers.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mlt.runstore.manifest import Manifest


_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    finished_at     TEXT,
    duration_s      REAL,
    fingerprint     TEXT,
    code_hash       TEXT,
    params_hash     TEXT,
    deps_hash       TEXT,
    python_version  TEXT,
    runner          TEXT,
    isolation       TEXT,
    metrics         TEXT,
    error           TEXT,
    verification_of TEXT,
    run_dir         TEXT NOT NULL
)
"""


class RunStore:
    """Manages runs for a single project."""

    def __init__(self, project_dir: Path) -> None:
        self._mlt_dir = project_dir / ".mlt"
        self._runs_dir = self._mlt_dir / "runs"
        self._db_path = self._mlt_dir / "index.sqlite"
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def runs_dir(self) -> Path:
        return self._runs_dir

    def run_dir(self, run_id: str) -> Path:
        return self._runs_dir / run_id

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def create_run(self, manifest: Manifest) -> Path:
        """Create the run directory, write the partial manifest and index it.

        Uses ``mkdir(exist_ok=False)`` as a lightweight mutex: two concurrent
        processes trying the same run_id will get an error (run_id includes
        random bytes, so collisions are astronomically rare).
        """
        run_dir = self._runs_dir / manifest.run_id
        run_dir.mkdir(parents=False, exist_ok=False)
        (run_dir / "model").mkdir()
        manifest.write(run_dir / "manifest.json")
        self._upsert(manifest)
        return run_dir

    def complete_run(self, manifest: Manifest) -> None:
        """Write the final manifest and update the index (atomic)."""
        run_dir = self._runs_dir / manifest.run_id
        manifest.write(run_dir / "manifest.json")
        self._upsert(manifest)

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark an existing run as failed and update the index."""
        path = self._runs_dir / run_id / "manifest.json"
        if path.exists():
            manifest = Manifest.read(path)
            manifest.mark_failed(error)
            manifest.write(path)
            self._upsert(manifest)

    def remove_run(self, run_id: str) -> None:
        """Delete run directory and remove from index (used when exit 3 inside env)."""
        import shutil

        run_dir = self._runs_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_run(self, run_id_or_prefix: str) -> Manifest:
        """Return the manifest for a run ID or unique prefix.

        Accepts the special alias ``latest``.
        Raises NotFoundError (exit 7) if not found or prefix is ambiguous.
        """
        run_id = self._resolve(run_id_or_prefix)
        path = self._runs_dir / run_id / "manifest.json"
        if not path.exists():
            from mlt.core.errors import NotFoundError

            raise NotFoundError(f"run manifest missing for {run_id}")
        return Manifest.read(path)

    def find_by_fingerprint(self, fingerprint: str) -> Manifest | None:
        """Return the first succeeded run with the given fingerprint, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id FROM runs WHERE fingerprint=? AND status='succeeded' LIMIT 1",
                (fingerprint,),
            ).fetchone()
        if row is None:
            return None
        return self.get_run(row[0])

    def list_runs(self, limit: int = 20, show_all: bool = False) -> list[Manifest]:
        """Return runs ordered by creation time (newest first)."""
        with self._connect() as conn:
            if show_all:
                rows = conn.execute(
                    "SELECT run_id FROM runs ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()

        result = []
        for (rid,) in rows:
            try:
                result.append(self.get_run(rid))
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_DDL)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _upsert(self, manifest: Manifest) -> None:
        metrics_json = (
            json.dumps(manifest.metrics) if manifest.metrics is not None else None
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, status, created_at, finished_at, duration_s,
                 fingerprint, code_hash, params_hash, deps_hash, python_version,
                 runner, isolation, metrics, error, verification_of, run_dir)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    manifest.run_id,
                    manifest.status,
                    manifest.created_at,
                    manifest.finished_at,
                    manifest.duration_s,
                    manifest.fingerprint,
                    manifest.inputs_code_hash,
                    manifest.inputs_params_hash,
                    manifest.inputs_deps_hash,
                    manifest.inputs_python_requested,
                    manifest.env_runner,
                    manifest.env_isolation,
                    metrics_json,
                    manifest.error,
                    manifest.verification_of,
                    str(self._runs_dir / manifest.run_id),
                ),
            )

    def _resolve(self, run_id_or_prefix: str) -> str:
        from mlt.core.errors import NotFoundError

        if run_id_or_prefix == "latest":
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            if row is None:
                raise NotFoundError("no runs found")
            return row[0]

        with self._connect() as conn:
            # Exact match
            row = conn.execute(
                "SELECT run_id FROM runs WHERE run_id=?", (run_id_or_prefix,)
            ).fetchone()
            if row:
                return row[0]
            # Prefix match
            rows = conn.execute(
                "SELECT run_id FROM runs WHERE run_id LIKE ?",
                (run_id_or_prefix + "%",),
            ).fetchall()

        if not rows:
            raise NotFoundError(f"run not found: {run_id_or_prefix!r}")
        if len(rows) > 1:
            options = ", ".join(r[0] for r in rows)
            raise NotFoundError(
                f"ambiguous run prefix {run_id_or_prefix!r}: matches {options}"
            )
        return rows[0][0]
