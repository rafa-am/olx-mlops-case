"""Tests for training failure modes.

Set A (fast) — mock ``train()`` and verify the CLI emits the correct exit code.
Set B (slow) — real venv; verify run-store state after each failure.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from mlt.cli.main import cli
from mlt.core.errors import ExecutionError, MltEnvironmentError, MltTimeoutError


# ---------------------------------------------------------------------------
# Set A: Fast CLI exit-code mapping (no venv, mock train())
# ---------------------------------------------------------------------------

# train_cmd imports ``train`` lazily (inside the function body), so we patch
# it at the source module.  We also need a minimal model.py so that
# require_project_dir() doesn't fail before train() is ever called.


def _stub_project(path: Path) -> None:
    """Write the minimum files needed to pass require_project_dir."""
    (path / "model.py").write_text("# stub\n")
    (path / "requirements.txt").write_text("scikit-learn>=1.3\n")
    (path / "data").mkdir()
    (path / "data" / "x.csv").write_text("x\n1\n")


def test_cli_train_exit_4_on_fit_exception(tmp_path, monkeypatch):
    """ExecutionError from trainer → exit 4."""
    _stub_project(tmp_path)
    monkeypatch.setattr(
        "mlt.core.trainer.train",
        lambda **_kw: (_ for _ in ()).throw(ExecutionError("fit raised RuntimeError")),
    )
    result = CliRunner().invoke(cli, ["train", str(tmp_path)])
    assert result.exit_code == 4


def test_cli_train_exit_5_on_timeout(tmp_path, monkeypatch):
    """MltTimeoutError from trainer → exit 5."""
    _stub_project(tmp_path)
    monkeypatch.setattr(
        "mlt.core.trainer.train",
        lambda **_kw: (_ for _ in ()).throw(MltTimeoutError("training timed out after 2s")),
    )
    result = CliRunner().invoke(cli, ["train", str(tmp_path)])
    assert result.exit_code == 5


def test_cli_train_exit_6_on_build_failure(tmp_path, monkeypatch):
    """MltEnvironmentError from trainer → exit 6."""
    _stub_project(tmp_path)
    monkeypatch.setattr(
        "mlt.core.trainer.train",
        lambda **_kw: (_ for _ in ()).throw(
            MltEnvironmentError("pip install failed: nonexistent-package-xyzzy not found")
        ),
    )
    result = CliRunner().invoke(cli, ["train", str(tmp_path)])
    assert result.exit_code == 6


def test_cli_train_exit_130_on_ctrl_c(tmp_path, monkeypatch):
    """KeyboardInterrupt from trainer → exit 130."""
    _stub_project(tmp_path)

    def _raise(**_kw):
        raise KeyboardInterrupt

    monkeypatch.setattr("mlt.core.trainer.train", _raise)
    result = CliRunner().invoke(cli, ["train", str(tmp_path)])
    assert result.exit_code == 130


# ---------------------------------------------------------------------------
# Set B: Slow integration tests (real venv, real subprocess)
# ---------------------------------------------------------------------------

pytestmark_slow = pytest.mark.slow

_FAIL_MODEL = """\
    from mlt.model.sklearn import SklearnModel

    class Model(SklearnModel):
        def fit(self, data_dir, params):
            raise RuntimeError("intentional failure in fit")
"""

_SLEEP_MODEL = """\
    import time
    from mlt.model.sklearn import SklearnModel

    class Model(SklearnModel):
        def fit(self, data_dir, params):
            time.sleep(300)
            return {}
"""

_REQ = "scikit-learn>=1.3\npandas>=2.0\nnumpy>=1.24\n"
_BAD_REQ = "nonexistent-package-xyzzy==0.0.1\n"


def _make_project(root: Path, model_src: str, requirements: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.py").write_text(textwrap.dedent(model_src))
    (root / "requirements.txt").write_text(requirements)
    data_dir = root / "data"
    data_dir.mkdir()
    csv = "x,y\n" + "\n".join(f"{i},{i % 2}" for i in range(10)) + "\n"
    (data_dir / "data.csv").write_text(csv)
    return {"project_dir": root, "data_dir": data_dir}


@pytest.mark.slow
def test_fit_exception_creates_failed_run(tmp_path):
    """fit() raising an exception → run status 'failed', error field set, log preserved."""
    from mlt.core.errors import ExecutionError
    from mlt.core.trainer import train
    from mlt.runstore.store import RunStore

    p = _make_project(tmp_path / "fail_proj", _FAIL_MODEL, _REQ)

    with pytest.raises(ExecutionError):
        train(p["project_dir"], data_dir=p["data_dir"])

    store = RunStore(p["project_dir"])
    runs = store.list_runs()
    assert len(runs) == 1
    m = runs[0]
    assert m.status == "failed"
    assert m.error is not None and len(m.error) > 0

    run_dir = store.run_dir(m.run_id)
    assert (run_dir / "train.log").exists()


@pytest.mark.slow
def test_timeout_creates_failed_run(tmp_path):
    """fit() sleeping past timeout → MltTimeoutError, run status 'failed'."""
    from mlt.core.errors import MltTimeoutError
    from mlt.core.trainer import train
    from mlt.runstore.store import RunStore

    p = _make_project(tmp_path / "sleep_proj", _SLEEP_MODEL, _REQ)

    with pytest.raises(MltTimeoutError) as exc_info:
        train(p["project_dir"], data_dir=p["data_dir"], timeout=2)

    assert "timeout" in str(exc_info.value.message).lower()

    store = RunStore(p["project_dir"])
    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0].status == "failed"
    assert "timeout" in (runs[0].error or "").lower()


@pytest.mark.slow
def test_build_failure_no_run_created(tmp_path):
    """Bad requirement → MltEnvironmentError before run creation → zero runs in store."""
    from mlt.core.errors import MltEnvironmentError
    from mlt.core.trainer import train
    from mlt.runstore.store import RunStore

    p = _make_project(tmp_path / "bad_req_proj", _FAIL_MODEL, _BAD_REQ)

    with pytest.raises(MltEnvironmentError):
        train(p["project_dir"], data_dir=p["data_dir"])

    store = RunStore(p["project_dir"])
    assert store.list_runs() == []
