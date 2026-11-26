"""mlt serve: start the inference API for a run (foreground)."""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import click

from mlt.cli.discovery import find_project_dir
from mlt.core.errors import MltEnvironmentError, OperationRefusedError
from mlt.runstore.store import RunStore

_HEALTH_TIMEOUT = 60  # seconds to wait for /health → 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_port(host: str, port: int) -> None:
    """Raise MltEnvironmentError if *port* is already bound on *host*."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            raise MltEnvironmentError(
                f"port {port} is already in use; use --port to specify another"
            )


def _example_value(spec: dict) -> Any:
    if "allowed" in spec:
        return spec["allowed"][0]
    t = spec.get("type", "")
    if t == "int":
        return int(spec.get("min", 0))
    if t == "float":
        return float(spec.get("min", 0.0))
    return ""


def _curl_example(run_id: str, schema: dict | None, host: str, port: int) -> str:
    """Generate a filled curl command from the run schema."""
    if schema and schema.get("features"):
        record = {name: _example_value(spec) for name, spec in schema["features"].items()}
    else:
        record = {"feature": "value"}
    body = json.dumps({"inputs": [record]})
    return (
        f'curl -s -X POST http://{host}:{port}/predict '
        f'-H "Content-Type: application/json" '
        f"-d '{body}'"
    )


def _load_schema(run_dir: Path) -> dict | None:
    path = run_dir / "schema_effective.yaml"
    if not path.exists():
        return None
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _venv_server_cmd(env_hash: str, run_dir: Path, host: str, port: int) -> list[str]:
    venv_dir = Path.home() / ".mlt" / "envs" / env_hash
    python = venv_dir / "bin" / "python"
    if not python.exists():
        raise MltEnvironmentError(
            f"venv for this run not found ({venv_dir}); re-train to rebuild the environment"
        )
    return [
        str(python), "-m", "mlt.serving.server", str(run_dir),
        "--host", host, "--port", str(port),
    ]


def _docker_server_cmd(env_hash: str, run_dir: Path, host: str, port: int) -> list[str]:
    image_tag = f"mlt-env:{env_hash[:12]}"
    # Inside the container: bind 0.0.0.0 so the port-forward reaches it.
    return [
        "docker", "run", "--rm",
        "-p", f"{host}:{port}:8000",
        "-v", f"{run_dir.resolve()}:/run:ro",
        image_tag,
        "python", "-m", "mlt.serving.server", "/run",
        "--host", "0.0.0.0", "--port", "8000",
    ]


def _wait_for_health(host: str, port: int, timeout: int = _HEALTH_TIMEOUT) -> None:
    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)
    raise MltEnvironmentError(
        f"server did not become healthy within {timeout}s; check the process output"
    )


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


@click.command("serve")
@click.argument("run_id", default="latest")
@click.option("--port", default=8000, show_default=True, type=int,
              help="Port to listen on.")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Host/IP to bind to.")
def serve_cmd(run_id: str, port: int, host: str) -> None:
    """Start the inference API for a run (foreground; Ctrl+C to stop)."""
    project_dir = find_project_dir()
    store = RunStore(project_dir)
    manifest = store.get_run(run_id)
    run_id = manifest.run_id  # resolve "latest" / prefix to concrete id

    if manifest.status != "succeeded":
        raise OperationRefusedError(
            f"run {run_id} has status '{manifest.status}'; "
            "only 'succeeded' runs can be served"
        )

    _check_port(host, port)

    run_dir = store.run_dir(run_id)
    schema = _load_schema(run_dir)

    env_runner = manifest.environment.get("runner", "venv")
    env_hash = manifest.environment.get("env_hash", "")

    if env_runner == "docker":
        from mlt.core.docker_check import docker_available
        if not docker_available():
            raise MltEnvironmentError(
                "run was trained with Docker but Docker daemon is not available"
            )
        cmd = _docker_server_cmd(env_hash, run_dir, host, port)
    else:
        cmd = _venv_server_cmd(env_hash, run_dir, host, port)

    click.echo(f"  run_id  {run_id}")
    click.echo(f"  runner  {env_runner}")
    click.echo("  starting server...")

    proc = subprocess.Popen(cmd)

    try:
        _wait_for_health(host, port)
    except MltEnvironmentError:
        proc.terminate()
        proc.wait()
        raise

    click.echo()
    click.echo(f"  url      http://{host}:{port}")
    click.echo(f"  docs     http://{host}:{port}/docs")
    click.echo()
    click.echo("  " + _curl_example(run_id, schema, host, port))
    click.echo()
    click.echo("  Ctrl+C to stop")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
