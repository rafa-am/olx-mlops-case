"""Hash functions for reproducibility fingerprinting.

The fingerprint is built from input hashes only (code, params, schema,
deps, python major.minor, seed). The runner/environment is excluded so
the same project trained in Docker and venv shares the same fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

# Directory names that are always excluded from the code hash.
_CODE_EXCLUDE = {".mlt", ".git", "__pycache__", ".venv"}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_str(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_dict(d: dict) -> str:
    """Canonical JSON hash of a dict (keys sorted, deterministic)."""
    return _sha256_str(json.dumps(d, sort_keys=True, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Input hash functions
# ---------------------------------------------------------------------------


def hash_code(project_dir: Path, extra_exclude: set[str] | None = None) -> str:
    """SHA-256 of all project files, excluding .mlt/, .git/, __pycache__, etc.

    ``extra_exclude`` adds names (e.g. the data directory) to the exclusion set.
    Files are sorted by relative path for determinism.
    """
    exclude = _CODE_EXCLUDE | (extra_exclude or set())

    file_hashes: dict[str, str] = {}
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir)
        if any(part in exclude for part in rel.parts):
            continue
        try:
            file_hashes[str(rel)] = _sha256_bytes(path.read_bytes())
        except (PermissionError, OSError):
            continue

    return _sha256_dict(file_hashes)


def hash_params(params: dict) -> str:
    """SHA-256 of params dict (canonical JSON, keys sorted)."""
    return _sha256_dict(params)


def hash_schema_file(schema_path: Path | None) -> str | None:
    """SHA-256 of a declared schema.yaml, or None if not present."""
    if schema_path is None or not schema_path.exists():
        return None
    return _sha256_bytes(schema_path.read_bytes())


def hash_deps(requirements_path: Path) -> str:
    """SHA-256 of normalized requirements.txt (sorted lines, no comments)."""
    text = requirements_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return _sha256_str("\n".join(sorted(lines)))


def hash_data_dir(data_dir: Path) -> dict:
    """Return per-file hashes and an aggregate hash for the data directory."""
    file_hashes: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(data_dir))
        file_hashes[rel] = _sha256_bytes(path.read_bytes())

    aggregate = _sha256_dict(file_hashes)
    return {"files": file_hashes, "hash": aggregate}


def mlt_source_hash() -> str:
    """Hash the installed mlt source tree.

    Included in the environment hash so that changes to mlt itself
    invalidate the cached venv.
    """
    import mlt

    mlt_dir = Path(mlt.__file__).parent
    return hash_code(mlt_dir)


def env_hash(deps_hash: str, python_minor: str) -> str:
    """Hash identifying a venv environment (deps + python version + mlt source).

    Runner is NOT included in the fingerprint but IS included in env_hash.
    """
    return _sha256_dict(
        {
            "deps": deps_hash,
            "python": python_minor,
            "mlt": mlt_source_hash(),
        }
    )


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def build_fingerprint(
    code_hash: str,
    params_hash: str,
    schema_hash: str | None,
    deps_hash: str,
    python_minor: str,
    seed: int,
) -> str:
    """Build the final fingerprint from all input hashes.

    The runner is intentionally excluded: the same project trained in
    Docker and venv (with the same Python) produces the same fingerprint.
    """
    return _sha256_dict(
        {
            "code": code_hash,
            "params": params_hash,
            "schema": schema_hash,
            "deps": deps_hash,
            "python": python_minor,
            "seed": seed,
        }
    )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def get_git_info(project_dir: Path) -> tuple[str | None, bool | None]:
    """Return (commit_sha, is_dirty) or (None, None) if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None, None
        commit = result.stdout.strip()

        dirty_result = subprocess.run(
            ["git", "-C", str(project_dir), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = bool(dirty_result.stdout.strip())
        return commit, dirty
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None


# ---------------------------------------------------------------------------
# Python version helpers
# ---------------------------------------------------------------------------


def resolve_python(
    project_dir: Path, python_flag: str | None
) -> tuple[str, str, str]:
    """Return (python_minor, python_requested, python_source).

    Precedence: --python flag > .python-version file > sys.executable version.
    For the venv runner, python_minor must match sys.executable; otherwise
    MltEnvironmentError is raised by the runner (not here).
    """
    import sys

    running_minor = ".".join(str(v) for v in sys.version_info[:2])

    if python_flag:
        return python_flag, python_flag, "flag"

    pv_file = project_dir / ".python-version"
    if pv_file.exists():
        requested = pv_file.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        return requested, requested, "file"

    return running_minor, running_minor, "default"
