"""Tests for mlt runs list and mlt runs show."""

import pytest
from click.testing import CliRunner

from mlt.cli.main import cli
from mlt.core.trainer import train
from mlt.runstore.store import RunStore


def invoke(*args):
    return CliRunner().invoke(cli, list(args))


# ---------------------------------------------------------------------------
# runs list
# ---------------------------------------------------------------------------


def test_runs_list_no_runs(tmp_path):
    """No runs → 'nenhum run ainda', exit 0."""
    import os
    old = os.getcwd()
    os.chdir(tmp_path)
    try:
        r = CliRunner().invoke(cli, ["runs", "list"])
    finally:
        os.chdir(old)
    assert r.exit_code == 0
    assert "nenhum run ainda" in r.output


def test_runs_list_shows_run(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])
    import os
    old_cwd = os.getcwd()
    os.chdir(p["project_dir"])
    try:
        r = CliRunner().invoke(cli, ["runs", "list"])
    finally:
        os.chdir(old_cwd)
    assert r.exit_code == 0, r.output
    assert run_id[:8] in r.output or run_id in r.output


def test_runs_list_shows_status(minimal_project):
    p = minimal_project
    train(p["project_dir"], data_dir=p["data_dir"])
    import os
    old_cwd = os.getcwd()
    os.chdir(p["project_dir"])
    try:
        r = CliRunner().invoke(cli, ["runs", "list"])
    finally:
        os.chdir(old_cwd)
    assert "succeeded" in r.output


# ---------------------------------------------------------------------------
# runs show
# ---------------------------------------------------------------------------


def test_runs_show_all_sections(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])

    import os
    old_cwd = os.getcwd()
    os.chdir(p["project_dir"])
    try:
        r = CliRunner().invoke(cli, ["runs", "show", run_id])
    finally:
        os.chdir(old_cwd)

    assert r.exit_code == 0, r.output
    assert run_id in r.output
    assert "ENTRADAS" in r.output
    assert "AMBIENTE" in r.output
    assert "RESULTADOS" in r.output
    assert "accuracy" in r.output


def test_runs_show_accepts_prefix(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])
    prefix = run_id[:8]

    import os
    old_cwd = os.getcwd()
    os.chdir(p["project_dir"])
    try:
        r = CliRunner().invoke(cli, ["runs", "show", prefix])
    finally:
        os.chdir(old_cwd)
    assert r.exit_code == 0, r.output


def test_runs_show_accepts_latest(minimal_project):
    p = minimal_project
    train(p["project_dir"], data_dir=p["data_dir"])

    import os
    old_cwd = os.getcwd()
    os.chdir(p["project_dir"])
    try:
        r = CliRunner().invoke(cli, ["runs", "show", "latest"])
    finally:
        os.chdir(old_cwd)
    assert r.exit_code == 0, r.output


def test_runs_show_nonexistent_fails(minimal_project):
    p = minimal_project
    import os
    old_cwd = os.getcwd()
    os.chdir(p["project_dir"])
    try:
        r = CliRunner().invoke(cli, ["runs", "show", "nonexistent-run-id"])
    finally:
        os.chdir(old_cwd)
    assert r.exit_code != 0
