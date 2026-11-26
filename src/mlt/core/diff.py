"""Core logic for comparing two runs.

Returns a structured RunDiff; never prints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlt.runstore.manifest import Manifest


def _get(d: dict, *keys, default=None):
    """Safely walk a nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)  # type: ignore[arg-type]
    return d


@dataclass
class RunDiff:
    """Structured result of comparing two runs."""

    fingerprint_match: bool
    # Which fingerprint components differ (empty when fingerprints match).
    fingerprint_components: list[str]

    # (key, value_in_a, value_in_b) for each differing param.
    param_diffs: list[tuple[str, Any, Any]] = field(default_factory=list)

    # Code hash diff: (hash_a, hash_b) when hashes differ, else None.
    code_hash_diff: tuple[str | None, str | None] | None = None
    # Commit diff: (commit_a, commit_b) when commits differ, else None.
    code_commit_diff: tuple[str | None, str | None] | None = None

    # Data hash diff: (hash_a, hash_b) when hashes differ, else None.
    data_hash_diff: tuple[str | None, str | None] | None = None
    # File count diff: (n_a, n_b) when counts differ, else None.
    data_files_diff: tuple[int, int] | None = None

    seed_diff: tuple[Any, Any] | None = None
    python_diff: tuple[Any, Any] | None = None
    deps_hash_diff: tuple[str | None, str | None] | None = None

    # (key, value_in_a, value_in_b) for each differing env field.
    env_diffs: list[tuple[str, Any, Any]] = field(default_factory=list)
    # (key, value_in_a, value_in_b) for each differing metric.
    metric_diffs: list[tuple[str, float | None, float | None]] = field(
        default_factory=list
    )

    @property
    def any_diff(self) -> bool:
        """True when at least one section has a difference to show."""
        return bool(
            self.param_diffs
            or self.code_hash_diff
            or self.data_hash_diff
            or self.seed_diff
            or self.python_diff
            or self.deps_hash_diff
            or self.env_diffs
            or self.metric_diffs
        )


def _fingerprint_components(m1: Manifest, m2: Manifest) -> list[str]:
    d1, d2 = m1.to_dict(), m2.to_dict()
    checks = [
        ("params",  ("inputs", "params", "hash")),
        ("code",    ("inputs", "code",   "hash")),
        ("data",    ("inputs", "data",   "hash")),
        ("seed",    ("inputs", "seed")),
        ("python",  ("inputs", "python", "requested")),
        ("deps",    ("inputs", "deps",   "hash")),
    ]
    return [
        label for label, path in checks
        if _get(d1, *path) != _get(d2, *path)
    ]


def diff_runs(m1: Manifest, m2: Manifest) -> RunDiff:
    """Compare two manifests and return a structured diff."""
    d1, d2 = m1.to_dict(), m2.to_dict()

    fp_match = m1.fingerprint == m2.fingerprint
    components = [] if fp_match else _fingerprint_components(m1, m2)

    result = RunDiff(
        fingerprint_match=fp_match,
        fingerprint_components=components,
    )

    # Params
    p1 = _get(d1, "inputs", "params", "values") or {}
    p2 = _get(d2, "inputs", "params", "values") or {}
    result.param_diffs = [
        (k, p1.get(k), p2.get(k))
        for k in sorted(set(p1) | set(p2))
        if p1.get(k) != p2.get(k)
    ]

    # Code
    c1 = _get(d1, "inputs", "code", "hash")
    c2 = _get(d2, "inputs", "code", "hash")
    if c1 != c2:
        result.code_hash_diff = (c1, c2)
        cm1 = _get(d1, "inputs", "code", "git_commit")
        cm2 = _get(d2, "inputs", "code", "git_commit")
        if cm1 != cm2:
            result.code_commit_diff = (cm1, cm2)

    # Data
    dh1 = _get(d1, "inputs", "data", "hash")
    dh2 = _get(d2, "inputs", "data", "hash")
    if dh1 != dh2:
        result.data_hash_diff = (dh1, dh2)
        f1 = len(_get(d1, "inputs", "data", "files") or {})
        f2 = len(_get(d2, "inputs", "data", "files") or {})
        if f1 != f2:
            result.data_files_diff = (f1, f2)

    # Seed
    s1 = _get(d1, "inputs", "seed")
    s2 = _get(d2, "inputs", "seed")
    if s1 != s2:
        result.seed_diff = (s1, s2)

    # Python
    py1 = _get(d1, "inputs", "python", "requested")
    py2 = _get(d2, "inputs", "python", "requested")
    if py1 != py2:
        result.python_diff = (py1, py2)

    # Deps
    dep1 = _get(d1, "inputs", "deps", "hash")
    dep2 = _get(d2, "inputs", "deps", "hash")
    if dep1 != dep2:
        result.deps_hash_diff = (dep1, dep2)

    # Environment (exclude internal env_hash key)
    env1 = d1.get("environment", {})
    env2 = d2.get("environment", {})
    env_keys = sorted(set(env1) | set(env2) - {"env_hash"})
    result.env_diffs = [
        (k, env1.get(k), env2.get(k))
        for k in env_keys
        if env1.get(k) != env2.get(k)
    ]

    # Metrics
    met1 = _get(d1, "outputs", "metrics") or {}
    met2 = _get(d2, "outputs", "metrics") or {}
    result.metric_diffs = [
        (k, met1.get(k), met2.get(k))
        for k in sorted(set(met1) | set(met2))
        if met1.get(k) != met2.get(k)
    ]

    return result


def diff_runs_by_id(
    project_dir: Path, run_id_a: str, run_id_b: str
) -> tuple[RunDiff, Manifest, Manifest]:
    """Resolve run IDs, load manifests and return (diff, m1, m2).

    Raises NotFoundError (exit 7) for unknown or ambiguous IDs.
    """
    from mlt.runstore.store import RunStore

    store = RunStore(project_dir)
    m1 = store.get_run(run_id_a)
    m2 = store.get_run(run_id_b)
    return diff_runs(m1, m2), m1, m2
