"""Runner interface: build an isolated environment and execute training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Runner(ABC):
    """Builds an isolated environment and runs the training worker inside it."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable runner name (e.g. 'venv', 'docker')."""

    @property
    @abstractmethod
    def isolation(self) -> str:
        """Isolation level recorded in the manifest (e.g. 'none', 'no-network')."""

    @abstractmethod
    def get_env_hash(self) -> str:
        """Hash identifying this environment (deps + python + mlt source).

        Used as the cache key; must NOT include the runner name so that
        a Docker and venv environment with the same deps share the fingerprint.
        """

    @abstractmethod
    def build_env(self, on_slow_build: "Callable[[], None] | None" = None) -> str:
        """Prepare the isolated environment and return the pip-freeze lock string.

        Raises MltEnvironmentError (exit 6) if the build fails.
        ``on_slow_build`` is called once if the build is known to take time
        (e.g. first Docker build) so the caller can print a progress hint.
        """

    @abstractmethod
    def run_training(
        self,
        run_dir: Path,
        project_dir: Path,
        data_dir: Path,
        seed: int,
        timeout: int = 600,
    ) -> int:
        """Execute the training worker and return its exit code.

        Writes stdout + stderr to ``run_dir/train.log``.
        """
