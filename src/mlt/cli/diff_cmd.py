"""mlt diff: compare two runs side by side."""

from __future__ import annotations

import click

from mlt.cli.discovery import find_project_dir


def _short(value, length: int = 8) -> str:
    if value is None:
        return "-"
    return str(value)[:length]


@click.command("diff")
@click.argument("run_a")
@click.argument("run_b")
def diff_cmd(run_a: str, run_b: str) -> None:
    """Compare two runs and show what changed."""
    from mlt.core.diff import diff_runs_by_id

    project_dir = find_project_dir()
    d, _m1, _m2 = diff_runs_by_id(project_dir, run_a, run_b)

    # --- Fingerprint summary ---
    if d.fingerprint_match:
        click.echo("fingerprint: " + click.style("identical", fg="green"))
    else:
        comp_str = ", ".join(d.fingerprint_components) if d.fingerprint_components else "unknown"
        click.echo(
            "fingerprint: " + click.style(f"different ({comp_str})", fg="yellow")
        )

    def _section(title: str) -> None:
        click.echo()
        click.echo(click.style(title, bold=True))

    # --- Params ---
    if d.param_diffs:
        _section("params")
        for k, v1, v2 in d.param_diffs:
            click.echo(f"  {k}  {v1} → {v2}")

    # --- Code ---
    if d.code_hash_diff:
        _section("code")
        c1, c2 = d.code_hash_diff
        click.echo(f"  hash    {_short(c1)} → {_short(c2)}")
        if d.code_commit_diff:
            cm1, cm2 = d.code_commit_diff
            click.echo(f"  commit  {_short(cm1)} → {_short(cm2)}")

    # --- Data ---
    if d.data_hash_diff:
        _section("data")
        dh1, dh2 = d.data_hash_diff
        click.echo(f"  hash   {_short(dh1)} → {_short(dh2)}")
        if d.data_files_diff:
            f1, f2 = d.data_files_diff
            click.echo(f"  files  {f1} → {f2}")

    # --- Seed ---
    if d.seed_diff:
        _section("seed")
        s1, s2 = d.seed_diff
        click.echo(f"  {s1} → {s2}")

    # --- Python ---
    if d.python_diff:
        _section("python")
        py1, py2 = d.python_diff
        click.echo(f"  {py1} → {py2}")

    # --- Deps ---
    if d.deps_hash_diff:
        _section("deps")
        dep1, dep2 = d.deps_hash_diff
        click.echo(f"  hash  {_short(dep1)} → {_short(dep2)}")

    # --- Environment ---
    if d.env_diffs:
        _section("environment")
        for k, v1, v2 in d.env_diffs:
            click.echo(f"  {k}  {v1} → {v2}")

    # --- Metrics ---
    if d.metric_diffs:
        _section("metrics")
        for k, v1, v2 in d.metric_diffs:
            if v1 is not None and v2 is not None:
                delta = v2 - v1
                sign = "+" if delta >= 0 else ""
                click.echo(f"  {k:<12}  {v1:.4f}  →  {v2:.4f}  (Δ {sign}{delta:.4f})")
            else:
                click.echo(f"  {k}  {v1} → {v2}")

    if not d.any_diff and d.fingerprint_match:
        click.echo()
        click.echo("identical inputs; no differences")
