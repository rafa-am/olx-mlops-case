"""Uvicorn entry point for the mlt Web UI."""

from __future__ import annotations

from pathlib import Path

import uvicorn

from mlt.ui.app import create_ui_app


def serve(project_dir: Path, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the UI server; blocks until Ctrl+C."""
    app = create_ui_app(project_dir)
    uvicorn.run(app, host=host, port=port, log_level="warning")
