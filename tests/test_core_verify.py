"""Unit tests for mlt.core.verify — no CliRunner."""

from __future__ import annotations

import pytest

from mlt.core.errors import OperationRefusedError
from mlt.core.trainer import train
from mlt.core.verify import VerifyResult, verify_run
from mlt.runstore.store import RunStore

pytestmark = pytest.mark.slow


def test_verify_run_returns_verify_result(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"], seed=42)
    result = verify_run(p["project_dir"], run_id, data_dir=p["data_dir"])
    assert isinstance(result, VerifyResult)


def test_verify_run_reproduced(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"], seed=42)
    result = verify_run(p["project_dir"], run_id, data_dir=p["data_dir"])
    assert result.reproduced
    assert result.fingerprint_match
    assert result.metric_diffs == []
    assert result.new_run_id != run_id


def test_verify_run_on_progress_called(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"], seed=42)
    messages: list[str] = []
    verify_run(
        p["project_dir"], run_id,
        data_dir=p["data_dir"],
        on_progress=messages.append,
    )
    assert any("retraining" in m for m in messages)
    assert any("runner" in m for m in messages)


def test_verify_run_refused_for_non_succeeded(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])
    RunStore(p["project_dir"]).fail_run(run_id, "forced")
    with pytest.raises(OperationRefusedError):
        verify_run(p["project_dir"], run_id, data_dir=p["data_dir"])


def test_verify_run_refused_when_data_changed(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"])
    (p["data_dir"] / "extra.csv").write_text("x\n1\n")
    with pytest.raises(OperationRefusedError):
        verify_run(p["project_dir"], run_id, data_dir=p["data_dir"])


def test_verify_run_result_carries_fingerprints(minimal_project):
    p = minimal_project
    run_id = train(p["project_dir"], data_dir=p["data_dir"], seed=42)
    result = verify_run(p["project_dir"], run_id, data_dir=p["data_dir"])
    assert result.orig_fingerprint is not None
    assert result.new_fingerprint is not None
    assert result.orig_fingerprint == result.new_fingerprint
