"""Tests for LocalDataSource."""

import pytest

from mlt.core.errors import ContractError
from mlt.datasources.local import LocalDataSource


def test_check_ok(tmp_path):
    (tmp_path / "data.csv").write_text("x\n1\n")
    LocalDataSource(tmp_path).check()  # must not raise


def test_check_nonexistent_dir(tmp_path):
    with pytest.raises(ContractError, match="not found"):
        LocalDataSource(tmp_path / "missing").check()


def test_check_empty_dir(tmp_path):
    with pytest.raises(ContractError, match="empty"):
        LocalDataSource(tmp_path).check()


def test_hash_files_returns_expected_structure(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    result = LocalDataSource(tmp_path).hash_files()
    assert "files" in result
    assert "hash" in result
    assert "a.csv" in result["files"]


def test_hash_files_deterministic(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    ds = LocalDataSource(tmp_path)
    assert ds.hash_files()["hash"] == ds.hash_files()["hash"]


def test_materialize_returns_path(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    ds = LocalDataSource(tmp_path)
    assert ds.materialize() == tmp_path
