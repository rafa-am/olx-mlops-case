"""Tests for mlt init command."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from mlt.cli.main import cli


def invoke(*args):
    return CliRunner().invoke(cli, list(args))


def test_init_creates_project_files(tmp_path):
    dest = tmp_path / "myproject"
    r = invoke("init", str(dest))
    assert r.exit_code == 0, r.output
    assert (dest / "model.py").exists()
    assert (dest / "requirements.txt").exists()
    assert (dest / "params.yaml").exists()


def test_init_creates_gitignore(tmp_path):
    dest = tmp_path / "myproject"
    invoke("init", str(dest))
    gitignore = (dest / ".gitignore").read_text()
    assert ".mlt/" in gitignore
    assert ".venv/" in gitignore
    assert "data/" in gitignore
    assert "__pycache__/" in gitignore


def test_init_prints_next_steps(tmp_path):
    dest = tmp_path / "proj"
    r = invoke("init", str(dest))
    assert "Next steps:" in r.output
    assert "mlt check" in r.output
    assert "mlt train" in r.output


def test_init_refuses_nonempty_dir(tmp_path):
    dest = tmp_path / "existing"
    dest.mkdir()
    (dest / "file.txt").write_text("x")
    r = invoke("init", str(dest))
    assert r.exit_code != 0
    assert "error:" in r.output or "not empty" in r.output


def test_init_accepts_dir_with_only_venv(tmp_path):
    """mlt init . after creating a local venv should succeed."""
    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / ".venv").mkdir()          # simulates: python3 -m venv .venv
    r = invoke("init", str(dest))
    assert r.exit_code == 0, r.output
    assert (dest / "model.py").exists()


def test_init_accepts_dir_with_only_git(tmp_path):
    dest = tmp_path / "proj"
    dest.mkdir()
    (dest / ".git").mkdir()
    r = invoke("init", str(dest))
    assert r.exit_code == 0, r.output


def test_init_accepts_empty_existing_dir(tmp_path):
    dest = tmp_path / "empty"
    dest.mkdir()
    r = invoke("init", str(dest))
    assert r.exit_code == 0
    assert (dest / "model.py").exists()


def test_init_creates_nonexistent_nested_dir(tmp_path):
    dest = tmp_path / "new" / "nested"
    r = invoke("init", str(dest))
    assert r.exit_code == 0
    assert dest.exists()
    assert (dest / "model.py").exists()


def test_init_default_template_is_iris(tmp_path):
    dest = tmp_path / "proj"
    r = invoke("init", str(dest))
    assert r.exit_code == 0
    # Iris template imports SklearnModel
    model_src = (dest / "model.py").read_text()
    assert "SklearnModel" in model_src


def test_init_copies_iris_csv(tmp_path):
    """iris template ships iris.csv so mlt train works immediately after init."""
    dest = tmp_path / "proj"
    invoke("init", str(dest))
    assert (dest / "data" / "iris.csv").exists()
    # 150 rows + header
    lines = (dest / "data" / "iris.csv").read_text().splitlines()
    assert len(lines) == 151
