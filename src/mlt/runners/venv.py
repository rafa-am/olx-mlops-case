"""Venv runner: creates a cached virtual environment and runs training inside it.

Environment cache: ``~/.mlt/envs/<env_hash>/``
Lock file:         ``~/.mlt/envs/<env_hash>.lock``  (fcntl.flock, exclusive)

The Python version is always ``sys.executable``: the venv runner does not
search the PATH for other Pythons.  If ``.python-version`` or ``--python``
requests a different version the caller (trainer) raises MltEnvironmentError
before the runner is instantiated.

``mlt`` is installed into the venv from its own source tree so that changes
to mlt itself are picked up (the env_hash includes the mlt source hash).
"""

from __future__ import annotations

import fcntl
import subprocess
import sys
from pathlib import Path
from typing import Callable

from mlt.runners.base import Runner

_ENVS_DIR = Path.home() / ".mlt" / "envs"




class VenvRunner(Runner):
    """Runner that creates a venv at ``~/.mlt/envs/<env_hash>/``."""

    def __init__(self, requirements_txt: Path, python_minor: str) -> None:
        self._requirements_txt = requirements_txt
        self._python_minor = python_minor
        self._python_exe = sys.executable
        self._env_hash_val: str | None = None
        self._lock_file = None

    # ------------------------------------------------------------------
    # Runner interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "venv"

    @property
    def isolation(self) -> str:
        return "none"

    def get_env_hash(self) -> str:
        if self._env_hash_val is None:
            from mlt.core.fingerprint import env_hash, hash_deps

            deps_h = hash_deps(self._requirements_txt)
            self._env_hash_val = env_hash(deps_h, self._python_minor)
        return self._env_hash_val

    def build_env(self, on_slow_build: Callable[[], None] | None = None) -> str:
        """Create or reuse the venv. Returns ``pip freeze`` output."""

        eh = self.get_env_hash()
        venv_dir = _ENVS_DIR / eh
        lock_path = _ENVS_DIR / f"{eh}.lock"
        _ENVS_DIR.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                if not _venv_ready(venv_dir):
                    if on_slow_build:
                        on_slow_build()
                    self._create_venv(venv_dir)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

        result = subprocess.run(
            [str(venv_dir / "bin" / "pip"), "freeze", "--all"],
            capture_output=True,
            text=True,
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
        """Run the training worker inside the venv. Returns the worker exit code."""
        eh = self.get_env_hash()
        venv_dir = _ENVS_DIR / eh
        python_bin = str(venv_dir / "bin" / "python")

        cmd = [
            python_bin,
            "-m",
            "mlt._worker.train",
            "--project-dir",
            str(project_dir),
            "--data-dir",
            str(data_dir),
            "--run-dir",
            str(run_dir),
            "--seed",
            str(seed),
        ]

        log_path = run_dir / "train.log"
        try:
            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    cwd=str(project_dir),
                )
            return result.returncode
        except subprocess.TimeoutExpired:
            # Return a sentinel; trainer maps this to MltTimeoutError.
            return _TIMEOUT_EXIT_CODE

    def run_deep_check(self, project_dir: Path, timeout: int = 60) -> tuple[int, str]:
        """Run the worker in validate-only mode inside the venv."""
        eh = self.get_env_hash()
        venv_dir = _ENVS_DIR / eh
        python_bin = str(venv_dir / "bin" / "python")

        cmd = [
            python_bin, "-m", "mlt._worker.train",
            "--project-dir", str(project_dir),
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

    def _create_venv(self, venv_dir: Path) -> None:
        from mlt.core.errors import MltEnvironmentError

        # Create the venv
        subprocess.run(
            [self._python_exe, "-m", "venv", str(venv_dir)],
            check=True,
            capture_output=True,
        )

        pip = str(venv_dir / "bin" / "pip")

        # Install mlt first so its dependencies are available inside the venv.
        # Editable install → pip install -e <source_root> (dev changes picked up).
        # Non-editable     → reconstruct a minimal layout in a temp dir and pip install.
        from mlt.runners._mlt_pkg import find_mlt_source_root, write_mlt_installable

        source_root = find_mlt_source_root()
        if source_root is not None:
            mlt_install_args = ["-e", str(source_root)]
            tmp_mlt_dir = None
        else:
            import tempfile
            tmp_mlt_dir = Path(tempfile.mkdtemp(prefix="mlt_venv_install_"))
            write_mlt_installable(tmp_mlt_dir)
            mlt_install_args = [str(tmp_mlt_dir)]

        try:
            result = subprocess.run(
                [pip, "install", "-q"] + mlt_install_args,
                capture_output=True,
                text=True,
            )
        finally:
            if tmp_mlt_dir is not None:
                import shutil as _shutil
                _shutil.rmtree(tmp_mlt_dir, ignore_errors=True)

        if result.returncode != 0:
            import shutil
            shutil.rmtree(venv_dir, ignore_errors=True)
            raise MltEnvironmentError(
                f"failed to install mlt into venv:\n{result.stderr[-500:]}"
            )

        # Install the scientist's requirements.
        result = subprocess.run(
            [pip, "install", "-q", "-r", str(self._requirements_txt)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            import shutil

            shutil.rmtree(venv_dir, ignore_errors=True)
            # Show the last 1 kB of pip output to help diagnose.
            excerpt = (result.stderr + result.stdout)[-1000:]
            raise MltEnvironmentError(
                f"build failed; install requirements failed:\n{excerpt}"
            )


# Sentinel exit code returned when the subprocess times out.
_TIMEOUT_EXIT_CODE = 124


def _venv_ready(venv_dir: Path) -> bool:
    return (venv_dir / "bin" / "python").exists() and (venv_dir / "bin" / "pip").exists()
