"""FastAPI application for the mlt local Web UI.

Passo 1 (read-only): Home, Runs, detail, Compare.

All routes are thin wrappers over mlt.core functions — same contract as the
CLI but returning JSON instead of formatted text.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import os
import subprocess
import sys
import threading
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
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


class _TrainState:
    """Thread-safe state for a single in-progress training job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.run_id: str | None = None
        self.status: str = "idle"   # idle | running | done | failed
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._cancel_event = threading.Event()
        self._runner = None  # set once train() creates the runner

    def start(
        self,
        project_dir: Path,
        runner: str,
        python: str | None,
        seed: int | None,
        env: dict,
    ) -> None:
        with self._lock:
            if self.status == "running":
                raise RuntimeError("A training job is already running.")
            self.run_id = None
            self.error = None
            self.status = "running"
            self._cancel_event.clear()

        def _run() -> None:
            try:
                from mlt.core.trainer import train

                run_id = train(
                    project_dir=project_dir,
                    data_dir=project_dir / "data",
                    runner=runner,
                    python=python,
                    seed=seed,
                    timeout=600,
                    force=False,
                    env=env,
                    on_runner_created=lambda r: setattr(self, "_runner", r),
                )
                with self._lock:
                    self.run_id = run_id
                    self.status = "done"
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
                    self.status = "failed"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Terminate the current subprocess and mark cancellation."""
        self._cancel_event.set()
        runner = self._runner
        if runner is not None:
            runner.terminate()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "run_id": self.run_id,
                "error":  self.error,
            }


def _not_found(exc: MltError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def create_ui_app(project_dir: Path) -> FastAPI:
    """Return a FastAPI app bound to *project_dir*."""
    app = FastAPI(title="mlt UI", docs_url=None, redoc_url=None)
    _train = _TrainState()

    # Reconcile any runs left in 'running' by a previous server session.
    from mlt.core.reconcile import reconcile_running_runs

    reconcile_running_runs(project_dir)

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

    @app.get("/api/build/log")
    def build_log(tail: int = 50):
        """Return the last *tail* lines of .mlt/build.log (live during env build)."""
        log_path = project_dir / ".mlt" / "build.log"
        if not log_path.exists():
            return {"lines": []}
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": lines[-tail:]}

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------

    def _read_params() -> dict:
        p = project_dir / "params.yaml"
        if not p.exists():
            return {}
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}

    @app.get("/api/project/params")
    def get_params():
        """Return the current params.yaml as a key-value dict."""
        return _read_params()

    @app.put("/api/project/params")
    async def put_params(request: Request):
        """Overwrite params.yaml with the provided key-value dict."""
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="Body must be a JSON object.")
        p = project_dir / "params.yaml"
        p.write_text(yaml.dump(body, default_flow_style=False), encoding="utf-8")
        return {"ok": True}

    # ------------------------------------------------------------------
    # Train (async)
    # ------------------------------------------------------------------

    @app.post("/api/train")
    async def start_train(request: Request):
        """Save params then launch training in a background thread."""
        body   = await request.json()
        params = body.get("params", {})
        runner = body.get("runner", "auto")
        python = body.get("python") or None   # e.g. "3.11", None → use default
        seed   = body.get("seed")             # int or None

        # Write params.yaml before training (matches CLI behaviour).
        p = project_dir / "params.yaml"
        p.write_text(yaml.dump(params, default_flow_style=False), encoding="utf-8")

        try:
            _train.start(
                project_dir=project_dir,
                runner=runner,
                python=python,
                seed=int(seed) if seed is not None else None,
                env=dict(os.environ),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"started": True}

    @app.get("/api/train/status")
    def train_status():
        """Return the current training state."""
        snap = _train.snapshot()
        # Reconcile: if the thread died unexpectedly while still marked running,
        # check whether the manifest already shows a terminal status.
        if snap["status"] == "running" and snap["run_id"]:
            try:
                m = describe_run(project_dir, snap["run_id"])
                if m.status in ("succeeded", "failed", "cancelled"):
                    with _train._lock:
                        _train.status = "done" if m.status == "succeeded" else "failed"
                    snap = _train.snapshot()
            except MltError:
                pass
        return snap

    @app.delete("/api/train")
    def cancel_train():
        """Request cancellation of the current training job (best-effort)."""
        snap = _train.snapshot()
        if snap["status"] != "running":
            raise HTTPException(status_code=409, detail="No training job in progress.")
        _train.cancel()
        # Mark the run as failed in the store if we have a run_id.
        if snap["run_id"]:
            try:
                from mlt.runstore.store import RunStore

                store = RunStore(project_dir)
                m = store.get_run(snap["run_id"])
                if m.status == "running":
                    m.mark_failed("interrupted by user (UI cancel)")
                    m.write(store.run_dir(m.run_id) / "manifest.json")
            except Exception:
                pass
        with _train._lock:
            _train.status = "failed"
            _train.error  = "Cancelled by user."
        return {"cancelled": True}

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
