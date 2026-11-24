"""Tests for MLModel and SklearnModel."""

import pytest
from pathlib import Path

from mlt.model.base import MLModel
from mlt.model.sklearn import SklearnModel


# ---------------------------------------------------------------------------
# Minimal concrete implementations used across tests
# ---------------------------------------------------------------------------


class _ConcreteModel(MLModel):
    def fit(self, data_dir: Path, params: dict) -> dict[str, float]:
        return {"metric": float(self.seed)}

    def predict(self, inputs: list[dict]) -> list:
        return [0] * len(inputs)

    def save(self, path: Path) -> None:
        (path / "marker.txt").write_text("saved")

    @classmethod
    def load(cls, path: Path) -> "_ConcreteModel":
        return cls()


class _SklearnConcreteModel(SklearnModel):
    def fit(self, data_dir: Path, params: dict) -> dict[str, float]:
        from sklearn.dummy import DummyClassifier

        self.model = DummyClassifier(strategy="most_frequent")
        self.model.fit([[0], [1]], [0, 1])
        return {"accuracy": 1.0}


# ---------------------------------------------------------------------------
# MLModel tests
# ---------------------------------------------------------------------------


def test_mlmodel_is_abstract():
    with pytest.raises(TypeError):
        MLModel()  # type: ignore[abstract]


def test_concrete_subclass_is_instantiable():
    model = _ConcreteModel()
    assert isinstance(model, MLModel)


def test_seed_is_injected_before_fit(tmp_path):
    model = _ConcreteModel()
    model.seed = 42
    metrics = model.fit(tmp_path, {})
    assert metrics["metric"] == 42.0


def test_fit_returns_dict_of_floats(tmp_path):
    model = _ConcreteModel()
    model.seed = 0
    result = model.fit(tmp_path, {})
    assert isinstance(result, dict)
    for v in result.values():
        assert isinstance(v, float)


def test_predict_returns_list(tmp_path):
    model = _ConcreteModel()
    model.seed = 0
    preds = model.predict([{"x": 1}, {"x": 2}])
    assert isinstance(preds, list)
    assert len(preds) == 2


def test_save_creates_file(tmp_path):
    model = _ConcreteModel()
    model.seed = 0
    model.save(tmp_path)
    assert (tmp_path / "marker.txt").exists()


# ---------------------------------------------------------------------------
# SklearnModel tests
# ---------------------------------------------------------------------------


def test_sklearn_fit_and_save_load_roundtrip(tmp_path):
    model = _SklearnConcreteModel()
    model.seed = 42
    metrics = model.fit(tmp_path, {})
    assert "accuracy" in metrics

    save_dir = tmp_path / "artifacts"
    save_dir.mkdir()
    model.save(save_dir)
    assert (save_dir / "model.joblib").exists()

    loaded = _SklearnConcreteModel.load(save_dir)
    assert hasattr(loaded, "model")
    preds = loaded.predict([{"x": 0}])
    assert len(preds) == 1


def test_sklearn_predict_without_model_raises():
    from mlt.core.errors import ExecutionError

    class _NeverFit(SklearnModel):
        def fit(self, data_dir, params):
            return {}

    model = _NeverFit()
    with pytest.raises(ExecutionError, match="self.model must be assigned"):
        model.predict([{"x": 0}])


def test_sklearn_save_without_model_raises(tmp_path):
    from mlt.core.errors import ExecutionError

    class _NeverFit(SklearnModel):
        def fit(self, data_dir, params):
            return {}

    model = _NeverFit()
    with pytest.raises(ExecutionError, match="self.model must be assigned"):
        model.save(tmp_path)


def test_sklearn_load_restores_predict_capability(tmp_path):
    """Round-trip through save/load; loaded model must predict correctly."""
    model = _SklearnConcreteModel()
    model.seed = 7
    model.fit(tmp_path, {})

    save_dir = tmp_path / "out"
    save_dir.mkdir()
    model.save(save_dir)

    loaded = _SklearnConcreteModel.load(save_dir)
    result = loaded.predict([{"a": 1, "b": 2}])
    assert isinstance(result, list)
    assert len(result) == 1
