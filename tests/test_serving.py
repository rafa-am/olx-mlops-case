"""Tests for the FastAPI serving app.

All tests use FastAPI's TestClient — no subprocess, no venv, no Docker.
The mock model is pure Python (no sklearn dependency), so these tests
run in the fast profile: ``pytest -m "not slow and not docker"``.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Mock model and fixtures
# ---------------------------------------------------------------------------


class _ConstantModel:
    """Mock model that returns a fixed label for every input."""

    def predict(self, inputs: list) -> list:
        return ["cat"] * len(inputs)


class _ErrorModel:
    """Mock model whose predict always raises."""

    def predict(self, inputs: list) -> list:
        raise ValueError("model broke")


def _make_manifest():
    from mlt.runstore.manifest import Manifest

    return Manifest.from_dict({
        "manifest_version": 1,
        "run_id": "20261125-120000-aabbcc",
        "status": "succeeded",
        "created_at": "2026-11-25T12:00:00+00:00",
        "finished_at": "2026-11-25T12:01:00+00:00",
        "duration_s": 60.0,
        "inputs": {
            "code": {"hash": "abc123"},
            "params": {"hash": "def456", "values": {}},
            "schema": {"hash": "ghi789"},
            "python": {"requested": "3.12", "source": "default"},
            "data": {
                "hash": "jkl012", "source": "local",
                "files": {}, "snapshot": None, "frozen": False,
            },
            "deps": {"hash": "mno345", "lock": []},
            "seed": 42,
            "seed_source": "default",
        },
        "fingerprint": "fp123456",
        "environment": {
            "runner": "venv",
            "isolation": "none",
            "python": "3.12.0",
            "env_hash": "envhash123",
        },
        "outputs": {
            "metrics": {"accuracy": 0.97},
            "schema": {"hash": "schema_hash_abc", "source": "inferred"},
            "artifacts": {},
        },
        "error": None,
        "verification_of": None,
    })


# Schema with two features: an int (with range) and a required string.
_SCHEMA = {
    "features": {
        "age": {"type": "int", "min": 0, "max": 120},
        "name": {"type": "str", "required": True},
    }
}


@pytest.fixture()
def app():
    from mlt.serving.app import create_app
    return create_app(_make_manifest(), _SCHEMA, _ConstantModel())


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture()
def client_no_schema():
    from mlt.serving.app import create_app
    from fastapi.testclient import TestClient
    return TestClient(create_app(_make_manifest(), None, _ConstantModel()))


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /metadata
# ---------------------------------------------------------------------------


def test_metadata(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    data = r.json()
    assert data["run_id"] == "20261125-120000-aabbcc"
    assert data["fingerprint"] == "fp123456"
    assert data["metrics"] == {"accuracy": 0.97}
    assert data["schema_hash"] == "schema_hash_abc"
    assert data["manifest_version"] == 1


# ---------------------------------------------------------------------------
# /schema
# ---------------------------------------------------------------------------


def test_schema_returns_features(client):
    r = client.get("/schema")
    assert r.status_code == 200
    body = r.json()
    assert "features" in body
    assert "age" in body["features"]
    assert "name" in body["features"]


def test_schema_no_schema_returns_empty(client_no_schema):
    r = client_no_schema.get("/schema")
    assert r.status_code == 200
    assert r.json() == {}


# ---------------------------------------------------------------------------
# /predict — happy path
# ---------------------------------------------------------------------------


def test_predict_basic_order(client):
    """Predictions count equals input count and run_id is present."""
    inputs = [
        {"age": 30, "name": "Alice"},
        {"age": 25, "name": "Bob"},
        {"age": 40, "name": "Carol"},
    ]
    r = client.post("/predict", json={"inputs": inputs})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 3
    assert body["run_id"] == "20261125-120000-aabbcc"


def test_predict_no_schema_accepts_any_record(client_no_schema):
    r = client_no_schema.post("/predict", json={"inputs": [{"anything": 1}]})
    assert r.status_code == 200


def test_predict_single_record(client):
    r = client.post("/predict", json={"inputs": [{"age": 10, "name": "X"}]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 1


# ---------------------------------------------------------------------------
# /predict — 400
# ---------------------------------------------------------------------------


def test_predict_missing_inputs_key(client):
    r = client.post(
        "/predict",
        content=b'{"data": []}',
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


def test_predict_empty_inputs_list(client):
    r = client.post("/predict", json={"inputs": []})
    assert r.status_code == 400


def test_predict_malformed_json(client):
    r = client.post(
        "/predict",
        content=b"{bad json}",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /predict — 413
# ---------------------------------------------------------------------------


def test_predict_batch_exceeds_limit(client_no_schema):
    from mlt.serving.app import BATCH_LIMIT

    inputs = [{"x": i} for i in range(BATCH_LIMIT + 1)]
    r = client_no_schema.post("/predict", json={"inputs": inputs})
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# /predict — 422 (schema validation)
# ---------------------------------------------------------------------------


def test_predict_missing_required_field(client):
    """Missing required 'name' → 422 with index + field in loc."""
    r = client.post("/predict", json={"inputs": [{"age": 30}]})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert any("name" in e["loc"] and "inputs[0]" in e["loc"] for e in detail)


def test_predict_wrong_type(client):
    """'age' must be int; passing a string → 422."""
    r = client.post("/predict", json={"inputs": [{"age": "not-an-int", "name": "Alice"}]})
    assert r.status_code == 422
    assert any("age" in e["loc"] for e in r.json()["detail"])


def test_predict_value_above_max(client):
    """age=200 exceeds max=120 → 422."""
    r = client.post("/predict", json={"inputs": [{"age": 200, "name": "Alice"}]})
    assert r.status_code == 422


def test_predict_value_below_min(client):
    """age=-1 is below min=0 → 422."""
    r = client.post("/predict", json={"inputs": [{"age": -1, "name": "Alice"}]})
    assert r.status_code == 422


def test_predict_multiple_records_one_invalid(client):
    """All-or-nothing: even if only one record is invalid, return 422."""
    inputs = [
        {"age": 30, "name": "Alice"},  # valid
        {"age": "bad", "name": "Bob"},  # invalid
    ]
    r = client.post("/predict", json={"inputs": inputs})
    assert r.status_code == 422
    # Error points to the second record.
    assert any("inputs[1]" in e["loc"] for e in r.json()["detail"])


# ---------------------------------------------------------------------------
# /predict — 500
# ---------------------------------------------------------------------------


def test_predict_model_exception_returns_500():
    """Exception inside predict → 500 without a stack trace in the body."""
    from mlt.serving.app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app(_make_manifest(), None, _ErrorModel()))
    r = client.post("/predict", json={"inputs": [{"x": 1}]})
    assert r.status_code == 500
    assert "Traceback" not in r.text


# ---------------------------------------------------------------------------
# validate_records unit tests
# ---------------------------------------------------------------------------


def test_validate_records_all_ok():
    from mlt.serving.app import validate_records

    errors = validate_records([{"age": 30, "name": "Alice"}], _SCHEMA)
    assert errors == []


def test_validate_records_allowed_values():
    from mlt.serving.app import validate_records

    schema = {"features": {"color": {"type": "str", "allowed": ["red", "green"]}}}
    errors = validate_records([{"color": "blue"}], schema)
    assert len(errors) == 1
    assert errors[0]["loc"] == "inputs[0].color"


def test_validate_records_optional_missing_field_ok():
    from mlt.serving.app import validate_records

    # 'age' is not required; missing it should not produce an error.
    errors = validate_records([{"name": "Alice"}], _SCHEMA)
    assert errors == []


def test_validate_records_multiple_errors_collected():
    from mlt.serving.app import validate_records

    # Both records are invalid; all errors are returned (not just the first).
    inputs = [{"age": "bad", "name": "A"}, {"age": 200, "name": "B"}]
    errors = validate_records(inputs, _SCHEMA)
    assert len(errors) == 2
    locs = {e["loc"] for e in errors}
    assert "inputs[0].age" in locs
    assert "inputs[1].age" in locs
