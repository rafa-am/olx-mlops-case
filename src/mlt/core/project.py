"""Core project validation.

``open_project`` is the single source of truth for what constitutes a valid
mlt project directory. Used by the CLI discovery layer and, eventually, by
the Web UI's project dialog.
"""

from __future__ import annotations

from pathlib import Path

from mlt.core.errors import UsageError


def open_project(path: Path | str) -> Path:
    """Validate *path* as an mlt project root and return its resolved Path.

    A valid project must contain ``model.py`` at the root level.

    Raises
    ------
    UsageError  If ``model.py`` is absent (exit 2).
    """
    resolved = Path(path).resolve()
    if not (resolved / "model.py").exists():
        raise UsageError(
            f"model.py not found in {resolved}. "
            "Pass the project directory as an argument or run from inside it."
        )
    return resolved
