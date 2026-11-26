"""mlt diff: compare two runs side by side."""

from __future__ import annotations

import click

from mlt.cli.discovery import find_project_dir
from mlt.runstore.manifest import Manifest


def _get(d: dict, *keys, default=None):
    """Safely walk a nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)  # type: ignore[arg-type]
    return d


def _fingerprint_components(m1: Manifest, m2: Manifest) -> list[str]:
    """Return which fingerprint components differ between two manifests."""
    d1 = m1.to_dict()
    d2 = m2.to_dict()
    checks = [
        ("params",  ("inputs", "params", "hash")),
        ("code",    ("inputs", "code",   "hash")),
        ("data",    ("inputs", "data",   "hash")),
        ("seed",    ("inputs", "seed")),
        ("python",  ("inputs", "python", "requested")),
        ("deps",    ("inputs", "deps",   "hash")),
    ]
    diffs = []
    for label, path in checks:
        v1 = _get(d1, *path)
        v2 = _get(d2, *path)
        if v1 != v2:
            diffs.append(label)
    return diffs


def _short(value, length: int = 8) -> str:
    if value is None:
        return "-"
    return str(value)[:length]


@click.command("diff")
@click.argument("run_a")
@click.argument("run_b")
def diff_cmd(run_a: str, run_b: str) -> None:
    """Compare two runs and show what changed."""
    from mlt.runstore.store import RunStore

    project_dir = find_project_dir()
    store = RunStore(project_dir)
    m1 = store.get_run(run_a)
    m2 = store.get_run(run_b)
    d1 = m1.to_dict()
    d2 = m2.to_dict()

    # --- Fingerprint summary ---
    if m1.fingerprint == m2.fingerprint:
        click.echo("fingerprint: " + click.style("identical", fg="green"))
    else:
        components = _fingerprint_components(m1, m2)
        comp_str = ", ".join(components) if components else "unknown"
        click.echo(
            "fingerprint: " + click.style(f"different ({comp_str})", fg="yellow")
        )

    any_diff = False

    def _section(title: str) -> None:
        nonlocal any_diff
        any_diff = True
        click.echo()
        click.echo(click.style(title, bold=True))

    # --- Params ---
    p1 = _get(d1, "inputs", "params", "values") or {}
    p2 = _get(d2, "inputs", "params", "values") or {}
    all_keys = sorted(set(p1) | set(p2))
    param_diffs = [(k, p1.get(k), p2.get(k)) for k in all_keys if p1.get(k) != p2.get(k)]
    if param_diffs:
        _section("params")
        for k, v1, v2 in param_diffs:
            click.echo(f"  {k}  {v1} → {v2}")

    # --- Code ---
    c1 = _get(d1, "inputs", "code", "hash")
    c2 = _get(d2, "inputs", "code", "hash")
    if c1 != c2:
        _section("code")
        click.echo(f"  hash    {_short(c1)} → {_short(c2)}")
        cm1 = _get(d1, "inputs", "code", "git_commit")
        cm2 = _get(d2, "inputs", "code", "git_commit")
        if cm1 != cm2:
            click.echo(f"  commit  {_short(cm1)} → {_short(cm2)}")

    # --- Data ---
    dh1 = _get(d1, "inputs", "data", "hash")
    dh2 = _get(d2, "inputs", "data", "hash")
    if dh1 != dh2:
        _section("data")
        click.echo(f"  hash   {_short(dh1)} → {_short(dh2)}")
        f1 = len(_get(d1, "inputs", "data", "files") or {})
        f2 = len(_get(d2, "inputs", "data", "files") or {})
        if f1 != f2:
            click.echo(f"  files  {f1} → {f2}")

    # --- Seed ---
    s1 = _get(d1, "inputs", "seed")
    s2 = _get(d2, "inputs", "seed")
    if s1 != s2:
        _section("seed")
        click.echo(f"  {s1} → {s2}")

    # --- Python ---
    py1 = _get(d1, "inputs", "python", "requested")
    py2 = _get(d2, "inputs", "python", "requested")
    if py1 != py2:
        _section("python")
        click.echo(f"  {py1} → {py2}")

    # --- Deps ---
    dep1 = _get(d1, "inputs", "deps", "hash")
    dep2 = _get(d2, "inputs", "deps", "hash")
    if dep1 != dep2:
        _section("deps")
        click.echo(f"  hash  {_short(dep1)} → {_short(dep2)}")

    # --- Environment ---
    env1 = d1.get("environment", {})
    env2 = d2.get("environment", {})
    env_keys = sorted(set(env1) | set(env2) - {"env_hash"})
    env_diffs = [(k, env1.get(k), env2.get(k)) for k in env_keys if env1.get(k) != env2.get(k)]
    if env_diffs:
        _section("environment")
        for k, v1, v2 in env_diffs:
            click.echo(f"  {k}  {v1} → {v2}")

    # --- Metrics ---
    met1 = _get(d1, "outputs", "metrics") or {}
    met2 = _get(d2, "outputs", "metrics") or {}
    all_metrics = sorted(set(met1) | set(met2))
    met_diffs = [
        (k, met1.get(k), met2.get(k))
        for k in all_metrics
        if met1.get(k) != met2.get(k)
    ]
    if met_diffs:
        _section("metrics")
        for k, v1, v2 in met_diffs:
            if v1 is not None and v2 is not None:
                delta = v2 - v1
                sign = "+" if delta >= 0 else ""
                click.echo(f"  {k:<12}  {v1:.4f}  →  {v2:.4f}  (Δ {sign}{delta:.4f})")
            else:
                click.echo(f"  {k}  {v1} → {v2}")

    if not any_diff and m1.fingerprint == m2.fingerprint:
        click.echo()
        click.echo("identical inputs; no differences")
