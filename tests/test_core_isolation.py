"""Verify that mlt.core does not import CLI or third-party UI frameworks.

This test grows with each Fase 1 extraction. It acts as a hard contract:
the core must remain usable by any frontend (CLI, Web UI, scripts) without
pulling in click, typer, or the CLI layer.
"""

from __future__ import annotations

import importlib
import sys


def _import_fresh(module_name: str) -> set[str]:
    """Import *module_name* in a subprocess-like way by temporarily removing
    it and its submodules from sys.modules, then re-importing.

    Returns the set of top-level package names that ended up in sys.modules
    as a result of the import.
    """
    # Save and strip any already-loaded mlt.core.* modules so we get a
    # clean import graph.
    saved = {k: v for k, v in sys.modules.items() if k == module_name or k.startswith(module_name + ".")}
    for k in saved:
        sys.modules.pop(k, None)

    before = set(sys.modules)
    importlib.import_module(module_name)
    after = set(sys.modules)

    # Restore original state.
    for k in list(sys.modules):
        if k not in before:
            sys.modules.pop(k, None)
    sys.modules.update(saved)

    return after - before


def _top_level(modules: set[str]) -> set[str]:
    return {m.split(".")[0] for m in modules}


def test_core_runs_does_not_import_click():
    imported = _top_level(_import_fresh("mlt.core.runs"))
    assert "click" not in imported, f"mlt.core.runs imported click via: {imported}"


def test_core_runs_does_not_import_typer():
    imported = _top_level(_import_fresh("mlt.core.runs"))
    assert "typer" not in imported


def test_core_runs_does_not_import_mlt_cli():
    imported = _import_fresh("mlt.core.runs")
    cli_modules = {m for m in imported if m.startswith("mlt.cli")}
    assert not cli_modules, f"mlt.core.runs imported CLI modules: {cli_modules}"
