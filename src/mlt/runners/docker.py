"""Docker runner: builds an image and runs training without network access.

Build context layout (temp dir):
  Dockerfile        — generated inline
  mlt_src/          — copy of the mlt source tree (pyproject.toml + src/)
  requirements.txt  — scientist's requirements (separate layer for caching)

Image tag: ``mlt-env:<env_hash[:12]>``

Training runs with ``--network none`` to match the design guarantee:
  "training runs without network".
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from mlt.runners.base import Runner

_IMAGE_PREFIX = "mlt-env"
_TIMEOUT_EXIT_CODE = 124


def _copy_mlt_to_context(ctx_path: Path) -> None:
    """Copy mlt into *ctx_path/mlt_src/* in a pip-installable layout.

    Editable install  → copy the whole source tree (pyproject.toml + src/).
    Non-editable      → reconstruct a minimal layout via write_mlt_installable.
    """
    from mlt.runners._mlt_pkg import find_mlt_source_root, write_mlt_installable

    source_root = find_mlt_source_root()
    if source_root is not None:
        shutil.copytree(
            source_root,
            ctx_path / "mlt_src",
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".git", ".venv", "*.egg-info",
            ),
        )
    else:
        write_mlt_installable(ctx_path / "mlt_src")


def _make_dockerfile(python_version: str) -> str:
    return (
        f"FROM python:{python_version}-slim\n"
        "WORKDIR /app\n"
        # mlt source first: layer is reused when only requirements.txt changes.
        "COPY mlt_src/ ./mlt_src/\n"
        "RUN pip install --no-cache-dir -q ./mlt_src\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir -q -r requirements.txt\n"
    )


class DockerRunner(Runner):
    """Runner that builds a Docker image and trains without network access."""

    def __init__(self, requirements_txt: Path, python_minor: str) -> None:
        self._requirements_txt = requirements_txt
        self._python_minor = python_minor
        self._env_hash_val: str | None = None

    # ------------------------------------------------------------------
    # Runner interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "docker"

    @property
    def isolation(self) -> str:
        return "no-network"

    def get_env_hash(self) -> str:
        if self._env_hash_val is None:
            from mlt.core.fingerprint import env_hash, hash_deps

            deps_h = hash_deps(self._requirements_txt)
            self._env_hash_val = env_hash(deps_h, self._python_minor)
        return self._env_hash_val

    def build_env(
        self,
        on_slow_build: Callable[[], None] | None = None,
        log_path: Path | None = None,
    ) -> str:
        """Build the Docker image if not cached. Returns pip freeze output."""
        from mlt.core.errors import MltEnvironmentError

        tag = self._image_tag()
        if not self._image_exists(tag):
            if on_slow_build:
                on_slow_build()
            self._build_image(tag, log_path=log_path)

        result = subprocess.run(
            ["docker", "run", "--rm", tag, "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise MltEnvironmentError(
                f"failed to get pip freeze from image:\n{result.stderr[-500:]}"
            )
        return result.stdout.strip()

    def run_training(
        self,
        run_dir: Path,
        project_dir: Path,
        data_dir: Path,
        seed: int,
        timeout: int = 600,
    ) -> int:
        """Run the worker inside the container with --network none."""
        tag = self._image_tag()
        log_path = run_dir / "train.log"

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "-v", f"{project_dir}:/project:ro",
            "-v", f"{data_dir}:/data:ro",
            "-v", f"{run_dir}:/run_dir",
            tag,
            "python", "-m", "mlt._worker.train",
            "--project-dir", "/project",
            "--data-dir", "/data",
            "--run-dir", "/run_dir",
            "--seed", str(seed),
        ]

        try:
            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            return result.returncode
        except subprocess.TimeoutExpired:
            return _TIMEOUT_EXIT_CODE

    def run_deep_check(self, project_dir: Path, timeout: int = 60) -> tuple[int, str]:
        """Run the worker in validate-only mode inside the container."""
        tag = self._image_tag()
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "-v", f"{project_dir}:/project:ro",
            tag,
            "python", "-m", "mlt._worker.train",
            "--project-dir", "/project",
            "--validate-only",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result.returncode, result.stderr
        except subprocess.TimeoutExpired:
            return _TIMEOUT_EXIT_CODE, "deep check timed out"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _image_tag(self) -> str:
        return f"{_IMAGE_PREFIX}:{self.get_env_hash()[:12]}"

    def _image_exists(self, tag: str) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
        )
        return result.returncode == 0

    def _build_image(self, tag: str, log_path: Path | None = None) -> None:
        from mlt.core.errors import MltEnvironmentError

        with tempfile.TemporaryDirectory() as ctx:
            ctx_path = Path(ctx)

            # Copy mlt into the build context (handles editable and non-editable).
            _copy_mlt_to_context(ctx_path)

            # requirements.txt as a separate layer for better caching.
            shutil.copy2(self._requirements_txt, ctx_path / "requirements.txt")

            (ctx_path / "Dockerfile").write_text(
                _make_dockerfile(self._python_minor), encoding="utf-8"
            )

            if log_path is None:
                result = subprocess.run(
                    ["docker", "build", "-t", tag, "."],
                    capture_output=True,
                    text=True,
                    cwd=ctx,
                )
                error_detail = (result.stderr + result.stdout)[-1000:]
            else:
                with open(log_path, "ab") as lf:
                    result = subprocess.run(
                        ["docker", "build", "-t", tag, "."],
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        cwd=ctx,
                    )
                if log_path.exists():
                    error_detail = "\n".join(
                        log_path.read_text(errors="replace").splitlines()[-30:]
                    )
                else:
                    error_detail = ""

        if result.returncode != 0:
            raise MltEnvironmentError(f"docker build failed:\n{error_detail}")
