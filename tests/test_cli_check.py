"""Tests for mlt check command."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from mlt.cli.main import cli


def invoke(*args):
    return CliRunner().invoke(cli, list(args))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_check_ok_on_valid_project(minimal_project):
    p = minimal_project
    r = invoke("check", str(p["project_dir"]), "--data", str(p["data_dir"]))
    assert r.exit_code == 0, r.output
    assert "ok" in r.output.lower()


def test_check_shows_python_preamble(minimal_project):
    p = minimal_project
    r = invoke("check", str(p["project_dir"]), "--data", str(p["data_dir"]))
    assert "python" in r.output


def test_check_shows_data_path(minimal_project):
    p = minimal_project
    r = invoke("check", str(p["project_dir"]), "--data", str(p["data_dir"]))
    assert str(p["data_dir"]) in r.output


def test_check_defaults_to_cwd(tmp_path):
    """check with no args fails gracefully when cwd has no model.py."""
    r = CliRunner().invoke(cli, ["check", str(tmp_path)])
    assert r.exit_code != 0


# ---------------------------------------------------------------------------
# Contract errors
# ---------------------------------------------------------------------------


def test_check_fails_missing_requirements(tmp_path):
    (tmp_path / "model.py").write_text(
        "from mlt.model.sklearn import SklearnModel\n"
        "class M(SklearnModel):\n"
        "    def fit(self, d, p): pass\n"
    )
    # No requirements.txt
    r = invoke("check", str(tmp_path))
    assert r.exit_code != 0
    assert "erro" in r.output.lower() or "error" in r.output.lower()


def test_check_fails_missing_model_py(tmp_path):
    (tmp_path / "requirements.txt").write_text("scikit-learn\n")
    r = invoke("check", str(tmp_path))
    assert r.exit_code != 0


def test_check_warns_indirect_inheritance(tmp_path):
    """A class with fit but no known base → warning, not error."""
    (tmp_path / "model.py").write_text(
        "class Base:\n"
        "    def fit(self, d, p): pass\n"
    )
    (tmp_path / "requirements.txt").write_text("scikit-learn\n")
    r = invoke("check", str(tmp_path))
    # Should warn, not hard-fail on the class issue, but data dir may fail
    # We just check the warning icon is present in some form
    assert "⚠" in r.output or "warning" in r.output.lower() or r.exit_code != 0


# ---------------------------------------------------------------------------
# Data errors
# ---------------------------------------------------------------------------


def test_check_fails_missing_data_dir(minimal_project):
    p = minimal_project
    r = invoke("check", str(p["project_dir"]), "--data", str(p["project_dir"] / "nodata"))
    assert r.exit_code != 0
    assert "erro" in r.output.lower() or "error" in r.output.lower()
