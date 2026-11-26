"""mlt runs: list and show runs."""

from __future__ import annotations

from datetime import datetime, timezone

import click

from mlt.cli.discovery import find_project_dir


def _age(created_at_iso: str) -> str:
    """Return a human-readable age string from an ISO timestamp."""
    try:
        created = datetime.fromisoformat(created_at_iso)
        delta = datetime.now(timezone.utc) - created
        s = int(delta.total_seconds())
        if s < 60:
            return f"{s}s ago"
        if s < 3600:
            return f"{s // 60}min ago"
        if s < 86400:
            return f"{s // 3600}h ago"
        return f"{s // 86400}d ago"
    except Exception:
        return created_at_iso[:16]


def _duration(duration_s: float | None) -> str:
    if duration_s is None:
        return "-"
    if duration_s < 60:
        return f"{duration_s:.1f}s"
    return f"{int(duration_s) // 60}m{int(duration_s) % 60}s"


def _short_metrics(metrics: dict | None, limit: int = 3) -> str:
    if not metrics:
        return "-"
    items = list(metrics.items())[:limit]
    return "  ".join(f"{k}={v:.4g}" for k, v in items)


@click.group("runs")
def runs_cmd() -> None:
    """List and inspect training runs."""


@runs_cmd.command("list")
@click.option("--all", "show_all", is_flag=True, default=False,
              help="Show all runs (default: last 20).")
def runs_list(show_all: bool) -> None:
    """List recent training runs."""
    from mlt.core.runs import list_runs

    project_dir = find_project_dir()
    runs = list_runs(project_dir, limit=20, show_all=show_all)

    if not runs:
        click.echo("no runs yet")
        return

    for m in runs:
        status_str = (
            click.style(m.status, fg="green")
            if m.status == "succeeded"
            else click.style(m.status, fg="red")
        )
        fp_short = (m.fingerprint or "")[:8] or "-"
        verify_mark = " [V]" if m.verification_of else ""
        metrics_str = _short_metrics(m.metrics)
        age_str = _age(m.created_at)
        dur_str = _duration(m.duration_s)
        click.echo(
            f"  {m.run_id}  {status_str}{verify_mark}"
            f"  {age_str}  {dur_str}  {metrics_str}  {fp_short}"
        )


@runs_cmd.command("show")
@click.argument("run_id")
def runs_show(run_id: str) -> None:
    """Show the details of a run."""
    from mlt.core.runs import describe_run
    from mlt.runstore.store import RunStore

    project_dir = find_project_dir()
    m = describe_run(project_dir, run_id)
    store = RunStore(project_dir)  # still needed for run_dir() in artifact path
    d = m.to_dict()

    # --- Identity ---
    status_str = (
        click.style(m.status, fg="green")
        if m.status == "succeeded"
        else click.style(m.status, fg="red")
    )
    click.echo(f"run_id    {m.run_id}")
    click.echo(f"status    {status_str}")
    click.echo(f"created   {m.created_at}")
    if m.finished_at:
        click.echo(f"finished  {m.finished_at}")
    click.echo(f"duration  {_duration(m.duration_s)}")
    if m.verification_of:
        click.echo(f"verifies  {m.verification_of}")

    # --- Inputs ---
    inp = d.get("inputs", {})
    if inp:
        click.echo()
        click.echo(click.style("INPUTS", bold=True))
        seed = inp.get("seed")
        seed_src = inp.get("seed_source", "")
        if seed is not None:
            click.echo(f"  seed    {seed}  ({seed_src})")
        py_info = inp.get("python", {})
        if py_info:
            click.echo(
                f"  python  {py_info.get('requested', '-')}  ({py_info.get('source', '-')})"
            )
        code_info = inp.get("code", {})
        if code_info:
            commit = code_info.get("git_commit") or "-"
            dirty = " (dirty)" if code_info.get("dirty") else ""
            click.echo(
                f"  code    {(code_info.get('hash') or '')[:8]}  commit: {commit[:8]}{dirty}"
            )
        params_info = inp.get("params", {})
        pvals = params_info.get("values", {})
        if pvals:
            pstr = "  ".join(f"{k}={v}" for k, v in pvals.items())
            click.echo(f"  params  {pstr}")
        data_info = inp.get("data", {})
        if data_info:
            files = data_info.get("files", {})
            click.echo(
                f"  data    {len(files)} file(s)  hash {(data_info.get('hash') or '')[:8]}"
            )
        deps_info = inp.get("deps", {})
        lock = deps_info.get("lock", [])
        click.echo(f"  deps    {len(lock)} package(s)")

    # --- Environment ---
    env = d.get("environment", {})
    if env:
        click.echo()
        click.echo(click.style("ENVIRONMENT", bold=True))
        click.echo(f"  runner     {env.get('runner', '-')}")
        click.echo(f"  isolation  {env.get('isolation', '-')}")
        if env.get("python"):
            click.echo(f"  python     {env['python']}")

    # --- Outputs ---
    out = d.get("outputs")
    if out:
        click.echo()
        click.echo(click.style("OUTPUTS", bold=True))
        for k, v in (out.get("metrics") or {}).items():
            click.echo(f"  {k:<10}  {v:.4f}")
        artifacts = out.get("artifacts", {})
        if artifacts:
            first = next(iter(artifacts))
            run_dir = store.run_dir(m.run_id)
            click.echo(f"  model      {run_dir / first}")

    # --- Error ---
    if m.error:
        click.echo()
        click.echo(click.style("ERROR", bold=True, fg="red"))
        click.echo(f"  {m.error}")
        log_path = store.run_dir(m.run_id) / "train.log"
        if log_path.exists():
            click.echo(f"  log: {log_path}")
