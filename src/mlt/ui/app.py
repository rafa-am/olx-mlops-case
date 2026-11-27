"""FastAPI application for the mlt local Web UI.

Passo 1 (read-only): Home, Runs, detail, Compare.

All routes are thin wrappers over mlt.core functions — same contract as the
CLI but returning JSON instead of formatted text.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from mlt.core.diff import diff_runs_by_id
from mlt.core.errors import MltError
from mlt.core.runs import describe_run, list_runs


def _docker_status() -> str:
    """Return 'available' or 'unavailable' (matches sidebar mock label)."""
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return "available" if r.returncode == 0 else "unavailable"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unavailable"


def _not_found(exc: MltError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def create_ui_app(project_dir: Path) -> FastAPI:
    """Return a FastAPI app bound to *project_dir*."""
    app = FastAPI(title="mlt UI", docs_url=None, redoc_url=None)

    # ------------------------------------------------------------------
    # API — JSON endpoints consumed by the SPA
    # ------------------------------------------------------------------

    @app.get("/api/meta")
    def meta():
        """Package metadata shown on the Home page."""
        md = importlib.metadata.metadata("mlt")
        return {
            "name": md["Name"],
            "version": md["Version"],
            "description": md["Summary"],
        }

    @app.get("/api/system")
    def system():
        """Host system info: docker status and server Python version."""
        return {
            "docker": _docker_status(),
            "python": sys.version.split()[0],
        }

    @app.get("/api/project")
    def project():
        """Project info: directory and existence of model.py."""
        return {
            "dir": str(project_dir),
            "has_model": (project_dir / "model.py").exists(),
        }

    @app.get("/api/project/settings")
    def project_settings():
        """Effective python, seed and runner for this project (no training)."""
        from mlt.core.settings import resolve_effective_settings

        try:
            s = resolve_effective_settings(project_dir)
            return {
                "python_requested": s.python_requested,
                "python_source":    s.python_source,
                "runner":           s.runner,
                "runner_display":   s.runner_display,
                "seed":             s.seed,
                "seed_source":      s.seed_source,
            }
        except Exception:
            # If settings can't be resolved (missing files etc.) return nulls.
            return {
                "python_requested": None,
                "python_source":    None,
                "runner":           None,
                "runner_display":   None,
                "seed":             None,
                "seed_source":      None,
            }

    @app.get("/api/runs")
    def runs(limit: int = 50, status: str | None = None):
        """List runs, newest first. Accepts optional *status* filter."""
        manifests = list_runs(project_dir, limit=limit, show_all=True)
        if status:
            manifests = [m for m in manifests if m.status == status]
        return [m.to_dict() for m in manifests]

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str):
        """Full manifest for a single run (accepts prefix and 'latest')."""
        try:
            return describe_run(project_dir, run_id).to_dict()
        except MltError as exc:
            raise _not_found(exc) from exc

    @app.get("/api/runs/{run_id}/log")
    def run_log(run_id: str, tail: int = 50):
        """Return the last *tail* lines of train.log for a run."""
        try:
            m = describe_run(project_dir, run_id)
        except MltError as exc:
            raise _not_found(exc) from exc

        from mlt.runstore.store import RunStore

        run_dir = RunStore(project_dir).run_dir(m.run_id)
        log_path = run_dir / "train.log"
        if not log_path.exists():
            return {"lines": []}
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": lines[-tail:]}

    @app.get("/api/diff/{run_id_a}/{run_id_b}")
    def diff(run_id_a: str, run_id_b: str):
        """Structured diff between two runs."""
        try:
            result, m1, m2 = diff_runs_by_id(project_dir, run_id_a, run_id_b)
        except MltError as exc:
            raise _not_found(exc) from exc

        # dataclasses.asdict converts tuples to lists, which is fine for JSON.
        d = dataclasses.asdict(result)
        d["run_id_a"] = m1.run_id
        d["run_id_b"] = m2.run_id
        return d

    # ------------------------------------------------------------------
    # SPA — serve index.html for every non-API path
    # ------------------------------------------------------------------

    _template = (Path(__file__).parent / "templates" / "index.html").read_text(
        encoding="utf-8"
    )

    @app.get("/", response_class=HTMLResponse)
    @app.get("/{path:path}", response_class=HTMLResponse)
    def index(path: str = ""):  # noqa: ARG001
        return _template

    return app
