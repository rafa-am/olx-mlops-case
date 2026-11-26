"""Convenience base for scikit-learn models.

Lazy imports: joblib and pandas are imported inside methods so this
module is importable without those packages installed on the host.
The scientist only needs to implement ``fit``, assigning ``self.model``.
"""

from __future__ import annotations

from pathlib import Path

from mlt.model.base import MLModel


class SklearnModel(MLModel):
    """MLModel convenience for scikit-learn.

    Implement ``fit`` and assign ``self.model`` to a fitted estimator.
    ``predict``, ``save``, and ``load`` are provided.

    Example::

        class Model(SklearnModel):
            def fit(self, data_dir, params):
                df = pd.read_csv(data_dir / "data.csv")
                self.model = RandomForestClassifier().fit(df[features], df[target])
                return {"accuracy": float(self.model.score(X_test, y_test))}
    """

    def predict(self, inputs: list[dict]) -> list:
        """Build a DataFrame from ``inputs`` and call ``self.model.predict``.

        Columns are reordered to match the training order using the model's
        ``feature_names_in_`` attribute (set by sklearn when trained with a
        named DataFrame).  This allows callers to pass features in any order.
        """
        import pandas as pd  # lazy: only needed at predict time

        if not hasattr(self, "model") or not hasattr(self.model, "predict"):
            from mlt.core.errors import ExecutionError

            raise ExecutionError(
                "self.model must be assigned during fit and expose a predict method"
            )
        df = pd.DataFrame(inputs)
        names = getattr(self.model, "feature_names_in_", None)
        if names is not None:
            df = df[list(names)]
        return list(self.model.predict(df))

    def save(self, path: Path) -> None:
        """Dump ``self.model`` to ``path/model.joblib`` using joblib."""
        import joblib  # lazy: only needed at save time

        if not hasattr(self, "model"):
            from mlt.core.errors import ExecutionError

            raise ExecutionError("self.model must be assigned during fit before save")
        joblib.dump(self.model, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> SklearnModel:
        """Load ``model.joblib`` from ``path`` and return a ready instance.

        ``cls`` must be the concrete subclass, not ``SklearnModel`` itself.
        """
        import joblib  # lazy: only needed at load time

        # Bypass __init__: the loaded model replaces construction.
        instance = cls.__new__(cls)
        instance.model = joblib.load(path / "model.joblib")
        return instance
