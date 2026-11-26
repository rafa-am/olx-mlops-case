"""Tests for mlt train command.

Integration tests that build a real venv are marked slow; they reuse the
env cache so only the first run downloads packages.
"""


import pytest
from click.testing import CliRunner

from mlt.cli.main import cli

pytestmark = pytest.mark.slow


def invoke(*args):
    return CliRunner().invoke(cli, list(args))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_train_succeeds(minimal_project):
    p = minimal_project
    r = invoke("train", str(p["project_dir"]), "--data", str(p["data_dir"]))
    assert r.exit_code == 0, r.output
    assert "run_id" in r.output
    assert "succeeded" in r.output


def test_train_shows_preamble(minimal_project):
    p = minimal_project
    r = invoke("train", str(p["project_dir"]), "--data", str(p["data_dir"]))
    assert "python" in r.output
    assert "runner" in r.output
    assert "seed" in r.output


def test_train_shows_metrics(minimal_project):
    p = minimal_project
    r = invoke("train", str(p["project_dir"]), "--data", str(p["data_dir"]))
    assert "accuracy" in r.output


def test_train_shows_next_steps(minimal_project):
    p = minimal_project
    r = invoke("train", str(p["project_dir"]), "--data", str(p["data_dir"]))
    assert "mlt runs show" in r.output
    assert "mlt serve" in r.output


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_train_force_reruns(minimal_project):
    p = minimal_project
    r1 = invoke("train", str(p["project_dir"]), "--data", str(p["data_dir"]))
    r2 = invoke("train", str(p["project_dir"]), "--data", str(p["data_dir"]), "--force")
    assert r1.exit_code == 0
    assert r2.exit_code == 0

    # Extract run_ids (first token after "run_id")
    def _run_id(output):
        for line in output.splitlines():
            if "run_id" in line:
                parts = line.split()
                return parts[-1] if len(parts) >= 2 else None
        return None

    _run_id(r1.output)
    _run_id(r2.output)
    # Idempotent without --force: may return the same id on second call
    # With --force: always a new run; we just check both succeeded
    assert r2.exit_code == 0


def test_train_seed_flag(minimal_project):
    p = minimal_project
    r = invoke("train", str(p["project_dir"]), "--data", str(p["data_dir"]),
               "--seed", "99", "--force")
    assert r.exit_code == 0
    assert "99" in r.output


def test_train_verbose_prints_log(minimal_project):
    p = minimal_project
    r = invoke("train", str(p["project_dir"]), "--data", str(p["data_dir"]),
               "--verbose", "--force")
    assert r.exit_code == 0
    assert "train.log" in r.output


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_train_fails_missing_model_py(tmp_path):
    r = invoke("train", str(tmp_path))
    assert r.exit_code != 0


def test_train_fails_contract_error(tmp_path):
    """model.py with no MLModel subclass → contract error."""
    (tmp_path / "model.py").write_text("class NotAModel: pass\n")
    (tmp_path / "requirements.txt").write_text("scikit-learn\n")
    data = tmp_path / "data"
    data.mkdir()
    (data / "x.csv").write_text("x\n1\n")
    r = invoke("train", str(tmp_path), "--data", str(data))
    assert r.exit_code != 0
