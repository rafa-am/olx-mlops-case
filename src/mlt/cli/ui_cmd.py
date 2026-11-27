"""CLI command: mlt ui — start the local Web UI."""

from __future__ import annotations

import click

from mlt.cli.discovery import require_project_dir


@click.command("ui")
@click.argument("project_dir", default=".", required=False)
@click.option("--port", default=8080, show_default=True, help="Port to listen on.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to.")
def ui_cmd(project_dir: str, port: int, host: str) -> None:
    """Start the local web UI for PROJECT_DIR (default: current directory)."""
    proj = require_project_dir(project_dir)

    # Lazy import keeps startup fast when the UI is not used.
    from mlt.ui.server import serve

    url = f"http://{host}:{port}"
    click.echo(f"mlt UI  {url}")
    click.echo(f"project {proj}")
    click.echo("Press Ctrl+C to stop.\n")
    serve(proj, host=host, port=port)
