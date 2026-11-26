"""Unit tests for mlt.core.settings and mlt.core.project."""

from __future__ import annotations

import pytest

from mlt.core.errors import UsageError
from mlt.core.project import open_project
from mlt.core.settings import EffectiveSettings, resolve_effective_settings


# ---------------------------------------------------------------------------
# open_project
# ---------------------------------------------------------------------------


def test_open_project_valid(tmp_path):
    (tmp_path / "model.py").touch()
    result = open_project(tmp_path)
    assert result == tmp_path.resolve()


def test_open_project_missing_model_py(tmp_path):
    with pytest.raises(UsageError):
        open_project(tmp_path)


def test_open_project_accepts_string(tmp_path):
    (tmp_path / "model.py").touch()
    result = open_project(str(tmp_path))
    assert result == tmp_path.resolve()


# ---------------------------------------------------------------------------
# resolve_effective_settings
# ---------------------------------------------------------------------------


def test_resolve_returns_effective_settings(tmp_path):
    (tmp_path / "model.py").touch()
    result = resolve_effective_settings(tmp_path)
    assert isinstance(result, EffectiveSettings)


def test_resolve_seed_from_flag(tmp_path):
    (tmp_path / "model.py").touch()
    result = resolve_effective_settings(tmp_path, seed_flag=99)
    assert result.seed == 99
    assert result.seed_source == "cli"


def test_resolve_seed_from_env(tmp_path):
    (tmp_path / "model.py").touch()
    result = resolve_effective_settings(tmp_path, env={"MLT_SEED": "7"})
    assert result.seed == 7
    assert result.seed_source == "env"


def test_resolve_seed_default(tmp_path):
    (tmp_path / "model.py").touch()
    result = resolve_effective_settings(tmp_path, env={})
    assert result.seed == 42
    assert result.seed_source == "default"


def test_resolve_python_from_flag(tmp_path):
    (tmp_path / "model.py").touch()
    result = resolve_effective_settings(tmp_path, python_flag="3.11")
    assert result.python_requested == "3.11"
    assert result.python_source == "flag"


def test_resolve_python_from_file(tmp_path):
    (tmp_path / "model.py").touch()
    (tmp_path / ".python-version").write_text("3.10\n")
    result = resolve_effective_settings(tmp_path)
    assert result.python_requested == "3.10"
    assert result.python_source == "file"


def test_resolve_python_default(tmp_path):
    (tmp_path / "model.py").touch()
    result = resolve_effective_settings(tmp_path, env={})
    assert result.python_source == "default"


def test_resolve_runner_explicit_venv(tmp_path):
    (tmp_path / "model.py").touch()
    result = resolve_effective_settings(tmp_path, runner="venv", env={})
    assert result.runner == "venv"
    assert result.runner_display == "venv"


def test_resolve_runner_auto_no_docker(tmp_path, monkeypatch):
    (tmp_path / "model.py").touch()
    monkeypatch.setattr("mlt.core.docker_check.docker_available", lambda: False)
    result = resolve_effective_settings(tmp_path, runner="auto", env={})
    assert result.runner == "venv"
    assert "auto" in result.runner_display


def test_resolve_runner_auto_with_docker(tmp_path, monkeypatch):
    (tmp_path / "model.py").touch()
    monkeypatch.setattr("mlt.core.docker_check.docker_available", lambda: True)
    result = resolve_effective_settings(tmp_path, runner="auto", env={})
    assert result.runner == "docker"
    assert "auto" in result.runner_display
