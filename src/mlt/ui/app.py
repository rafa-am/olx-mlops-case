"""FastAPI application for the mlt local Web UI.

Passo 1 (read-only): Home, Runs, detail, Compare.

All routes are thin wrappers over mlt.core functions — same contract as the
CLI but returning JSON instead of formatted text.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from mlt.core.diff import diff_runs_by_id
from mlt.core.errors import MltError
from mlt.core.runs import describe_run, list_runs

# Single source of truth for defaults shared with the frontend via /api/system.
# Keep in sync with CLI defaults in train_cmd.py and serve_cmd.py.
_DEFAULT_PORT = 8000
_DEFAULT_TIMEOUT = 600
_RUNNERS = ["auto", "venv", "docker"]


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
        force: bool = False,
        timeout: int = _DEFAULT_TIMEOUT,
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
                    timeout=timeout,
                    force=force,
                    env=env,
                    on_runner_created=lambda r: setattr(self, "_runner", r),
                )
                with self._lock:
                    # Don't overwrite an explicit cancel (status already "failed").
                    if self.status != "failed":
                        self.run_id = run_id
                        self.status = "done"
            except Exception as exc:
                with self._lock:
                    # Preserve the user-facing "Cancelled by user." message if
                    # the exception was triggered by our own terminate() call.
                    if self.status != "failed":
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


class _ServeState:
    """State for a single inference service process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.run_id: str | None = None
        self.host: str = "127.0.0.1"
        self.port: int | None = None
        self.status: str = "idle"   # idle | starting | running | failed
        self.error: str | None = None
        self._proc: subprocess.Popen | None = None

    def start(self, run_id: str, cmd: list, host: str, port: int) -> None:
        with self._lock:
            if self.status in ("starting", "running"):
                raise RuntimeError("A service is already running.")
            self.run_id = run_id
            self.host = host
            self.port = port
            self.status = "starting"
            self.error = None

        self._proc = subprocess.Popen(cmd)

        def _wait() -> None:
            url = f"http://{host}:{port}/health"
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                proc = self._proc
                if proc is None or proc.poll() is not None:
                    with self._lock:
                        self.status = "failed"
                        self.error = "server exited unexpectedly during startup"
                    return
                try:
                    with urllib.request.urlopen(url, timeout=2) as r:
                        if r.status == 200:
                            with self._lock:
                                self.status = "running"
                            return
                except Exception:
                    pass
                time.sleep(0.5)
            proc = self._proc
            if proc:
                proc.terminate()
                proc.wait()
            with self._lock:
                self.status = "failed"
                self.error = "server did not become healthy within 60s"

        threading.Thread(target=_wait, daemon=True).start()

    def stop(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        with self._lock:
            self.status = "idle"
            self.run_id = None
            self.port = None
            self.error = None
            self._proc = None

    def snapshot(self) -> dict:
        with self._lock:
            if self.status == "running" and self._proc and self._proc.poll() is not None:
                self.status = "failed"
                self.error = f"server exited with code {self._proc.returncode}"
            return {
                "status": self.status,
                "run_id": self.run_id,
                "host": self.host,
                "port": self.port,
                "url": f"http://{self.host}:{self.port}" if self.port else None,
                "error": self.error,
            }


class _VerifyState:
    """Thread-safe state for a single in-progress verification job."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.run_id: str | None = None        # run being verified
        self.new_run_id: str | None = None    # new run created by verify
        self.status: str = "idle"             # idle | running | done | failed
        self.error: str | None = None
        self.result: dict | None = None       # VerifyResult fields on completion
        self._thread: threading.Thread | None = None
        self._runner = None

    def start(self, project_dir: Path, run_id: str, runner: str | None = None) -> None:
        with self._lock:
            if self.status == "running":
                raise RuntimeError("A verification job is already running.")
            self.run_id = run_id
            self.new_run_id = None
            self.error = None
            self.result = None
            self.status = "running"

        def _run() -> None:
            try:
                from mlt.core.verify import verify_run
                r = verify_run(
                    project_dir=project_dir,
                    run_id=run_id,
                    runner=runner,
                    on_runner_created=lambda r: setattr(self, "_runner", r),
                )
                with self._lock:
                    if self.status != "failed":
                        self.new_run_id = r.new_run_id
                        self.result = {
                            "reproduced":        r.reproduced,
                            "fingerprint_match": r.fingerprint_match,
                            "metric_diffs":      r.metric_diffs,
                            "orig_fingerprint":  r.orig_fingerprint,
                            "new_fingerprint":   r.new_fingerprint,
                            "effective_runner":  r.effective_runner,
                            "effective_python":  r.effective_python,
                            "effective_seed":    r.effective_seed,
                        }
                        self.status = "done"
            except Exception as exc:
                with self._lock:
                    if self.status != "failed":
                        self.error = str(exc)
                        self.status = "failed"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        runner = self._runner
        if runner is not None:
            runner.terminate()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status":     self.status,
                "run_id":     self.run_id,
                "new_run_id": self.new_run_id,
                "error":      self.error,
                "result":     self.result,
            }


def _not_found(exc: MltError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def create_ui_app(project_dir: Path) -> FastAPI:
    """Return a FastAPI app bound to *project_dir*."""
    app = FastAPI(title="mlt UI", docs_url=None, redoc_url=None)
    _train = _TrainState()
    _verify = _VerifyState()
    _serve = _ServeState()

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
        """Host system info, UI defaults and available runners."""
        return {
            "docker":          _docker_status(),
            "python":          sys.version.split()[0],
            "runners":         _RUNNERS,
            "default_port":    _DEFAULT_PORT,
            "default_timeout": _DEFAULT_TIMEOUT,
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
        force  = bool(body.get("force", False))

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
                force=force,
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

    # ------------------------------------------------------------------
    # Verify (reproducibility check)
    # ------------------------------------------------------------------

    @app.post("/api/verify/{run_id}")
    async def start_verify(run_id: str, request: Request):
        """Start a verification job for a succeeded run."""
        ct   = request.headers.get("content-type", "")
        body = await request.json() if "application/json" in ct else {}
        runner = body.get("runner") or None  # None → use original run's runner
        try:
            _verify.start(project_dir, run_id, runner=runner)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"started": True}

    @app.get("/api/verify/status")
    def verify_status():
        """Return the current verification state."""
        return _verify.snapshot()

    @app.delete("/api/verify")
    def cancel_verify():
        """Request cancellation of the current verification job (best-effort)."""
        snap = _verify.snapshot()
        if snap["status"] != "running":
            raise HTTPException(status_code=409, detail="No verification job in progress.")
        _verify.cancel()
        with _verify._lock:
            _verify.status = "failed"
            _verify.error  = "Cancelled by user."
        return {"cancelled": True}

    # ------------------------------------------------------------------
    # Serve (inference API process)
    # ------------------------------------------------------------------

    def _build_serve_cmd(manifest, run_dir: Path, host: str, port: int) -> list:
        env_runner = manifest.environment.get("runner", "venv")
        env_hash = manifest.environment.get("env_hash", "")
        if env_runner == "docker":
            return [
                "docker", "run", "--rm",
                "-p", f"{host}:{port}:8000",
                "-v", f"{run_dir.resolve()}:/run:ro",
                f"mlt-env:{env_hash[:12]}",
                "python", "-m", "mlt.serving.server", "/run",
                "--host", "0.0.0.0", "--port", "8000",
            ]
        venv_dir = Path.home() / ".mlt" / "envs" / env_hash
        python = venv_dir / "bin" / "python"
        if not python.exists():
            raise HTTPException(
                status_code=409,
                detail=(
                    f"venv for this run not found ({venv_dir}); "
                    "re-train to rebuild the environment"
                ),
            )
        return [
            str(python), "-m", "mlt.serving.server", str(run_dir),
            "--host", host, "--port", str(port),
        ]

    @app.get("/api/serve/status")
    def serve_status():
        """Current serve process state, with schema when running."""
        from mlt.runstore.store import RunStore

        snap = _serve.snapshot()
        schema: dict | None = None
        if snap["status"] == "running" and snap["run_id"]:
            try:
                m = describe_run(project_dir, snap["run_id"])
                run_dir = RunStore(project_dir).run_dir(m.run_id)
                p = run_dir / "schema_effective.yaml"
                schema = yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}
            except Exception:
                schema = {}
        snap["schema"] = schema
        return snap

    @app.post("/api/serve")
    async def start_serve(request: Request):
        """Start an inference service for a succeeded run."""
        from mlt.runstore.store import RunStore

        body    = await request.json()
        run_id  = body.get("run_id")
        port    = int(body.get("port", 8000))
        host    = body.get("host", "127.0.0.1")

        try:
            m = describe_run(project_dir, run_id)
        except MltError as exc:
            raise _not_found(exc) from exc

        if m.status != "succeeded":
            raise HTTPException(
                status_code=409,
                detail=f"run {m.run_id} has status '{m.status}'; only succeeded runs can be served",
            )

        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
            except OSError:
                raise HTTPException(status_code=409, detail=f"port {port} is already in use")

        run_dir = RunStore(project_dir).run_dir(m.run_id)
        cmd = _build_serve_cmd(m, run_dir, host, port)

        try:
            _serve.start(m.run_id, cmd, host, port)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return {"started": True}

    @app.delete("/api/serve")
    def stop_serve():
        """Stop the active inference service."""
        snap = _serve.snapshot()
        if snap["status"] == "idle":
            raise HTTPException(status_code=409, detail="no active service")
        _serve.stop()
        return {"stopped": True}

    @app.post("/api/serve/predict")
    async def serve_predict(request: Request):
        """Proxy a predict request to the running inference service."""
        snap = _serve.snapshot()
        if snap["status"] != "running":
            raise HTTPException(status_code=409, detail="no active service")
        body = await request.body()
        req = urllib.request.Request(
            f"{snap['url']}/predict",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return JSONResponse(content=json.loads(r.read()), status_code=200)
        except urllib.error.HTTPError as exc:
            return JSONResponse(content=json.loads(exc.read()), status_code=exc.code)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"predict failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Init (new project)
    # ------------------------------------------------------------------

    _TEMPLATE_FILES = {
        "iris": [
            ("model.py",           "MLModel class (SklearnModel) ready for iris"),
            ("requirements.txt",   "model dependencies"),
            ("params.yaml",        "hyperparameters (optional)"),
            (".python-version",    "pinned Python version"),
            (".gitignore",         "excludes .mlt/, data/, .venv/"),
            ("data/iris.csv",      "sample data"),
        ],
    }

    @app.get("/api/templates")
    def list_templates():
        """Return available templates and their file lists."""
        return {
            name: [{"name": n, "description": d} for n, d in files]
            for name, files in _TEMPLATE_FILES.items()
        }

    @app.post("/api/init")
    async def init_project(request: Request):
        """Create a new project from a template (equivalent to mlt init)."""
        import shutil as _shutil

        body      = await request.json()
        directory = body.get("directory", "").strip()
        template  = body.get("template", "iris")

        if not directory:
            raise HTTPException(status_code=422, detail="directory is required")
        if template not in _TEMPLATE_FILES:
            raise HTTPException(status_code=422, detail=f"unknown template '{template}'")

        dest = Path(directory).expanduser().resolve()

        if dest.exists() and any(not p.name.startswith(".") for p in dest.iterdir()):
            raise HTTPException(
                status_code=409,
                detail=f"{dest} already exists and is not empty",
            )

        template_dir = Path(__file__).parent.parent / "templates" / template
        dest.mkdir(parents=True, exist_ok=True)

        for src in sorted(template_dir.rglob("*")):
            if src.is_file() and "__pycache__" not in src.parts:
                rel = src.relative_to(template_dir)
                dst = dest / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                _shutil.copy2(src, dst)

        return {
            "directory": str(dest),
            "files": [{"name": n, "description": d} for n, d in _TEMPLATE_FILES[template]],
        }

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
