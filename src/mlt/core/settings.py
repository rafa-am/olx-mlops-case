"""Resolve effective training settings from flags, files and environment.

Returns structured data; never prints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass
class EffectiveSettings:
    """Resolved values and their origins for a training run."""

    # Python
    python_minor: str      # actual major.minor used (e.g. "3.10")
    python_requested: str  # value shown to the user
    python_source: str     # "flag" | "file" | "default"

    # Seed
    seed: int
    seed_source: str       # "cli" | "env" | "default"

    # Runner
    runner: str            # resolved runner: "docker" | "venv"
    runner_display: str    # display string: "docker" | "venv" | "auto → venv"


def resolve_effective_settings(
    project_dir: Path,
    python_flag: str | None = None,
    seed_flag: int | None = None,
    runner: str = "auto",
    env: Mapping[str, str] | None = None,
) -> EffectiveSettings:
    """Resolve python, seed and runner from flags, project files and environment.

    Parameters
    ----------
    project_dir:  Project root (used to find ``.python-version``).
    python_flag:  Value of ``--python``; ``None`` means not provided.
    seed_flag:    Value of ``--seed``; ``None`` means not provided.
    runner:       ``"auto"``, ``"venv"``, or ``"docker"``.
    env:          Environment mapping; defaults to ``os.environ``.
    """
    if env is None:
        env = os.environ

    from mlt.core.docker_check import docker_available
    from mlt.core.fingerprint import resolve_python
    from mlt.core.seed import resolve_seed

    python_minor, python_requested, python_source = resolve_python(
        project_dir, python_flag
    )
    seed_val, seed_origin = resolve_seed(seed_flag, env)

    if runner == "auto":
        effective_runner = "docker" if docker_available() else "venv"
        runner_display = f"auto → {effective_runner}"
    else:
        effective_runner = runner
        runner_display = runner

    return EffectiveSettings(
        python_minor=python_minor,
        python_requested=python_requested,
        python_source=python_source,
        seed=seed_val,
        seed_source=seed_origin,
        runner=effective_runner,
        runner_display=runner_display,
    )
