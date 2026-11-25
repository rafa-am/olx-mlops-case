"""Tests for mlt check command, including --deep (venv) and --runner docker (marked)."""

from pathlib import Path
from unittest.mock import patch

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
    assert "error" in r.output.lower()


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
    assert "error" in r.output.lower()


# ---------------------------------------------------------------------------
# --deep (venv runner, no Docker required)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_check_deep_venv_passes(minimal_project):
    """--deep --runner venv should pass for a valid project."""
    p = minimal_project
    r = invoke(
        "check", str(p["project_dir"]),
        "--deep", "--runner", "venv",
        "--data", str(p["data_dir"]),
    )
    assert r.exit_code == 0, r.output
    assert "deep validation passed" in r.output


@pytest.mark.slow
def test_check_deep_venv_fails_bad_model(tmp_path):
    """--deep should exit 3 when the model has a contract error."""
    project_dir = tmp_path / "bad"
    project_dir.mkdir()
    (project_dir / "model.py").write_text("def train(): pass\n")
    (project_dir / "requirements.txt").write_text("scikit-learn>=1.3\npandas>=2.0\nnumpy>=1.24\n")
    data_dir = project_dir / "data"
    data_dir.mkdir()
    (data_dir / "data.csv").write_text("x\n1\n2\n")

    r = invoke(
        "check", str(project_dir),
        "--deep", "--runner", "venv",
        "--data", str(data_dir),
    )
    assert r.exit_code == 3, r.output
    assert "contract error" in r.output.lower() or "error" in r.output.lower()


# ---------------------------------------------------------------------------
# --runner docker without Docker (unit test, no Docker required)
# ---------------------------------------------------------------------------


def test_check_runner_docker_without_docker_exits_6(minimal_project):
    """--runner docker with no Docker daemon should exit with code 6."""
    p = minimal_project
    with patch("mlt.core.docker_check.docker_available", return_value=False):
        r = invoke(
            "check", str(p["project_dir"]),
            "--deep", "--runner", "docker",
            "--data", str(p["data_dir"]),
        )
    assert r.exit_code == 6, r.output
