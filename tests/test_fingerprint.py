"""Tests for fingerprint and hash functions."""



from mlt.core.fingerprint import (
    build_fingerprint,
    hash_code,
    hash_data_dir,
    hash_deps,
    hash_params,
    hash_schema_file,
)

# ---------------------------------------------------------------------------
# hash_code
# ---------------------------------------------------------------------------


def test_hash_code_deterministic(tmp_path):
    (tmp_path / "model.py").write_text("class Model: pass\n")
    h1 = hash_code(tmp_path)
    h2 = hash_code(tmp_path)
    assert h1 == h2


def test_hash_code_changes_on_file_edit(tmp_path):
    f = tmp_path / "model.py"
    f.write_text("class Model: pass\n")
    h1 = hash_code(tmp_path)
    f.write_text("class Model: pass\n# changed\n")
    h2 = hash_code(tmp_path)
    assert h1 != h2


def test_hash_code_ignores_mlt_dir(tmp_path):
    (tmp_path / "model.py").write_text("class Model: pass\n")
    h1 = hash_code(tmp_path)

    mlt_dir = tmp_path / ".mlt"
    mlt_dir.mkdir()
    (mlt_dir / "index.sqlite").write_bytes(b"\x00\x01")
    h2 = hash_code(tmp_path)
    assert h1 == h2


def test_hash_code_ignores_pycache(tmp_path):
    (tmp_path / "model.py").write_text("class Model: pass\n")
    h1 = hash_code(tmp_path)

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "model.cpython-310.pyc").write_bytes(b"\x00\x01")
    h2 = hash_code(tmp_path)
    assert h1 == h2


def test_hash_code_respects_extra_exclude(tmp_path):
    (tmp_path / "model.py").write_text("class Model: pass\n")
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text("x,y\n1,2\n")

    h_with_data = hash_code(tmp_path)
    h_without_data = hash_code(tmp_path, extra_exclude={"data"})
    assert h_with_data != h_without_data

    # Adding more data files does not affect hash when excluded.
    (data / "extra.csv").write_text("a,b\n3,4\n")
    h_still_excluded = hash_code(tmp_path, extra_exclude={"data"})
    assert h_without_data == h_still_excluded


# ---------------------------------------------------------------------------
# hash_params
# ---------------------------------------------------------------------------


def test_hash_params_empty_is_stable():
    h1 = hash_params({})
    h2 = hash_params({})
    assert h1 == h2


def test_hash_params_different_values():
    h1 = hash_params({"n": 100})
    h2 = hash_params({"n": 200})
    assert h1 != h2


def test_hash_params_key_order_independent():
    h1 = hash_params({"a": 1, "b": 2})
    h2 = hash_params({"b": 2, "a": 1})
    assert h1 == h2


# ---------------------------------------------------------------------------
# hash_deps
# ---------------------------------------------------------------------------


def test_hash_deps_sorted_order_independent(tmp_path):
    r1 = tmp_path / "r1.txt"
    r2 = tmp_path / "r2.txt"
    r1.write_text("scikit-learn>=1.3\npandas>=2.0\n")
    r2.write_text("pandas>=2.0\nscikit-learn>=1.3\n")
    assert hash_deps(r1) == hash_deps(r2)


def test_hash_deps_ignores_comments(tmp_path):
    r1 = tmp_path / "r1.txt"
    r2 = tmp_path / "r2.txt"
    r1.write_text("scikit-learn>=1.3\n")
    r2.write_text("# this is a comment\nscikit-learn>=1.3\n")
    assert hash_deps(r1) == hash_deps(r2)


# ---------------------------------------------------------------------------
# hash_data_dir
# ---------------------------------------------------------------------------


def test_hash_data_dir_structure(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    result = hash_data_dir(tmp_path)
    assert "files" in result
    assert "hash" in result
    assert "a.csv" in result["files"]


def test_hash_data_dir_deterministic(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    r1 = hash_data_dir(tmp_path)
    r2 = hash_data_dir(tmp_path)
    assert r1["hash"] == r2["hash"]


# ---------------------------------------------------------------------------
# build_fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_same_inputs_same_hash():
    fp1 = build_fingerprint("ch", "ph", None, "dh", "3.10", 42)
    fp2 = build_fingerprint("ch", "ph", None, "dh", "3.10", 42)
    assert fp1 == fp2


def test_fingerprint_different_seed():
    fp1 = build_fingerprint("ch", "ph", None, "dh", "3.10", 42)
    fp2 = build_fingerprint("ch", "ph", None, "dh", "3.10", 99)
    assert fp1 != fp2


def test_fingerprint_different_params():
    fp1 = build_fingerprint("ch", "ph1", None, "dh", "3.10", 42)
    fp2 = build_fingerprint("ch", "ph2", None, "dh", "3.10", 42)
    assert fp1 != fp2


def test_fingerprint_different_python():
    fp1 = build_fingerprint("ch", "ph", None, "dh", "3.10", 42)
    fp2 = build_fingerprint("ch", "ph", None, "dh", "3.11", 42)
    assert fp1 != fp2


def test_fingerprint_schema_included(tmp_path):
    schema = tmp_path / "schema.yaml"
    schema.write_text("features:\n  x: {type: float}\n")
    h = hash_schema_file(schema)

    fp_with = build_fingerprint("ch", "ph", h, "dh", "3.10", 42)
    fp_without = build_fingerprint("ch", "ph", None, "dh", "3.10", 42)
    assert fp_with != fp_without
