"""Project directory discovery for CLI commands.

``require_project_dir``: used by ``train`` and ``check``, which receive an
explicit path and require ``model.py`` to be present there.

``find_project_dir``: used by ``runs``, ``diff`` and ``verify``, which walk up
from the current directory until they find ``model.py`` or ``.mlt/``, then fall
back to ``.`` so RunStore can respond "no runs yet" gracefully.
"""

from __future__ import annotations

from pathlib import Path

from mlt.core.errors import UsageError


def require_project_dir(path: str | Path = ".") -> Path:
    """Resolve an explicit project path; raise UsageError if model.py is absent."""
    resolved = Path(path).resolve()
    if not (resolved / "model.py").exists():
        raise UsageError(
            f"model.py not found in {resolved}. "
            "Pass the project directory as an argument or run from inside it."
        )
    return resolved


def find_project_dir(start: str | Path = ".") -> Path:
    """Walk up from *start* to find a project root (``model.py`` or ``.mlt/``).

    Falls back to the resolved *start* path so RunStore can handle the
    "no runs yet" case gracefully instead of this function failing.
    """
    current = Path(start).resolve()
    while True:
        if (current / "model.py").exists() or (current / ".mlt").is_dir():
            return current
        parent = current.parent
        if parent == current:
            # Hit the filesystem root without finding a project.
            return Path(start).resolve()
        current = parent
