"""mlt init: create a new project from a template."""

from __future__ import annotations

import shutil
from pathlib import Path

import click

from mlt.core.errors import UsageError


_GITIGNORE = """\
.mlt/
data/
__pycache__/
*.pyc
*.pyo
*.egg-info/
"""


@click.command("init")
@click.argument("directory")
@click.option(
    "--template",
    default="iris",
    show_default=True,
    type=click.Choice(["iris"]),
    help="Project template to use.",
)
def init_cmd(directory: str, template: str) -> None:
    """Create a new mlt project from a template."""
    dest = Path(directory).resolve()

    if dest.exists() and any(dest.iterdir()):
        raise UsageError(
            f"{dest} already exists and is not empty. "
            "Remove it or choose a different directory."
        )

    template_dir = Path(__file__).parent.parent / "templates" / template
    if not template_dir.is_dir():
        raise UsageError(f"template '{template}' not found in package")

    dest.mkdir(parents=True, exist_ok=True)

    for src in sorted(template_dir.rglob("*")):
        if src.is_file():
            rel = src.relative_to(template_dir)
            dst = dest / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Always generate a fresh .gitignore (may overwrite a template one).
    (dest / ".gitignore").write_text(_GITIGNORE, encoding="utf-8")

    click.echo(f"Projeto criado em {dest}")
    click.echo()
    click.echo("Próximos passos:")
    click.echo(f"  cd {directory}")
    click.echo("  mlt check")
    click.echo("  mlt train")
