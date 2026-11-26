"""Utilities to locate and package mlt for installation inside isolated envs.

Both VenvRunner and DockerRunner need to install mlt into their target
environments.  The strategy differs depending on how mlt itself is installed
on the host:

Editable install  (dev / source tree present)
  ``find_mlt_source_root()`` returns the directory that contains
  ``pyproject.toml`` + ``src/``.  Callers can run ``pip install [-e] <root>``.

Non-editable install  (end-user, installed via pip into a venv)
  No source tree exists.  ``write_mlt_installable(dst)`` reconstructs a
  minimal pip-installable layout under *dst*:
      dst/
        pyproject.toml   (generated, mirrors runtime deps)
        src/mlt/         (copy of the installed package)
  Callers can then run ``pip install <dst>``.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def find_mlt_source_root() -> Path | None:
    """Return the source tree root if mlt is installed in editable mode, else None."""
    import mlt

    candidate = Path(mlt.__file__).parent  # the mlt/ package directory
    for _ in range(6):
        candidate = candidate.parent
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate
    return None


def _installed_requires() -> list[str]:
    """Return the runtime dependencies declared in mlt's installed metadata."""
    from importlib.metadata import requires

    deps = requires("mlt") or []
    # importlib.metadata includes extras like 'click>=8.0 ; extra == "dev"'.
    # Keep only the unconditional ones (no '; extra ==' marker).
    return [d for d in deps if "; extra ==" not in d]


def write_mlt_installable(dst: Path) -> None:
    """Write a minimal pip-installable mlt layout under *dst*.

    Used when no source tree is available (non-editable host install).
    After this call ``pip install <dst>`` works without network access.
    Runtime dependencies are read from the installed package metadata so
    this stays correct when pyproject.toml changes.
    """
    import mlt as mlt_module

    mlt_pkg = Path(mlt_module.__file__).parent  # .../site-packages/mlt/
    shutil.copytree(
        mlt_pkg,
        dst / "src" / "mlt",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    deps_lines = "".join(f'    "{d}",\n' for d in _installed_requires())
    (dst / "pyproject.toml").write_text(
        "[build-system]\n"
        'requires = ["setuptools>=61"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        "[project]\n"
        'name = "mlt"\n'
        f'version = "{mlt_module.__version__}"\n'
        f"dependencies = [\n{deps_lines}]\n\n"
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n',
        encoding="utf-8",
    )
