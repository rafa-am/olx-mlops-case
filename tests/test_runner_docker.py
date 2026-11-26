"""Tests for the Docker runner.

All tests in this module require Docker and are skipped automatically
when the Docker daemon is not available.

Run only these tests with:
  pytest -m docker
"""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from mlt.cli.main import cli
from mlt.core.trainer import train

pytestmark = pytest.mark.docker


def _invoke_from(project_dir, *args):
    old = os.getcwd()
    os.chdir(project_dir)
    try:
        return CliRunner().invoke(cli, list(args))
    finally:
        os.chdir(old)


# ---------------------------------------------------------------------------
# DockerRunner unit-level tests
# ---------------------------------------------------------------------------


def test_docker_runner_builds_image(iris_project):
    """build_env should produce a Docker image on first call."""
    from mlt.runners.docker import DockerRunner

    p = iris_project
    runner = DockerRunner(p["project_dir"] / "requirements.txt", "3.12")
    tag = runner._image_tag()

    # Clean up any pre-existing image to test a fresh build.
    import subprocess
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)

    lock = runner.build_env()
    assert runner._image_exists(tag)
    assert "scikit-learn" in lock.lower()


def test_docker_runner_caches_image(iris_project):
    """Second build_env call reuses the cached image (on_slow_build not called)."""
    from mlt.runners.docker import DockerRunner

    p = iris_project
    runner = DockerRunner(p["project_dir"] / "requirements.txt", "3.12")

    # Ensure the image exists first.
    runner.build_env()

    slow_called = []
    runner.build_env(on_slow_build=lambda: slow_called.append(True))
    assert not slow_called, "on_slow_build should not be called when image is cached"


def test_docker_runner_deep_check_passes(iris_project):
    """run_deep_check should return exit code 0 for a valid project."""
    from mlt.runners.docker import DockerRunner

    p = iris_project
    runner = DockerRunner(p["project_dir"] / "requirements.txt", "3.12")
    runner.build_env()

    exit_code, stderr = runner.run_deep_check(p["project_dir"])
    assert exit_code == 0, f"deep check failed:\n{stderr}"


def test_docker_runner_deep_check_fails_bad_model(tmp_path):
    """run_deep_check should return exit code 3 for a contract violation."""
    from mlt.runners.docker import DockerRunner

    project_dir = tmp_path / "bad"
    project_dir.mkdir()
    # Model with no MLModel subclass.
    (project_dir / "model.py").write_text("def train(): pass\n")
    (project_dir / "requirements.txt").write_text("scikit-learn>=1.3\npandas>=2.0\nnumpy>=1.24\n")

    runner = DockerRunner(project_dir / "requirements.txt", "3.12")
    runner.build_env()

    exit_code, stderr = runner.run_deep_check(project_dir)
    assert exit_code == 3
    assert "contract error" in stderr.lower()


# ---------------------------------------------------------------------------
# Integration: train with Docker runner
# ---------------------------------------------------------------------------


def test_train_docker_succeeds(iris_project):
    """Full training run with --runner docker should succeed."""
    p = iris_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"], runner="docker")
    assert run_id


def test_train_docker_same_fingerprint_as_venv(iris_project):
    """Docker and venv produce the same fingerprint (runner is not in fingerprint)."""
    from mlt.runstore.store import RunStore

    p = iris_project
    seed = 7

    venv_id = train(p["project_dir"], data_dir=p["data_dir"], runner="venv", seed=seed, force=True)
    docker_id = train(
        p["project_dir"], data_dir=p["data_dir"], runner="docker", seed=seed, force=True
    )

    store = RunStore(p["project_dir"])
    venv_fp = store.get_run(venv_id).fingerprint
    docker_fp = store.get_run(docker_id).fingerprint
    assert venv_fp == docker_fp, "fingerprints differ between venv and docker runs"


def test_train_docker_records_isolation(iris_project):
    """Docker run manifest should record isolation: no-network."""
    from mlt.runstore.store import RunStore

    p = iris_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"], runner="docker")
    store = RunStore(p["project_dir"])
    manifest = store.get_run(run_id)
    assert manifest.environment["isolation"] == "no-network"


# ---------------------------------------------------------------------------
# CLI: mlt train --runner docker
# ---------------------------------------------------------------------------


def test_cli_train_docker(iris_project):
    p = iris_project
    r = _invoke_from(p["project_dir"], "train", str(p["project_dir"]), "--runner", "docker")
    assert r.exit_code == 0, r.output
    assert "succeeded" in r.output


# ---------------------------------------------------------------------------
# CLI: mlt check --deep --runner docker
# ---------------------------------------------------------------------------


def test_check_deep_docker_passes(iris_project):
    p = iris_project
    r = _invoke_from(
        p["project_dir"],
        "check", str(p["project_dir"]),
        "--deep", "--runner", "docker",
        "--data", str(p["data_dir"]),
    )
    assert r.exit_code == 0, r.output
    assert "deep validation passed" in r.output
