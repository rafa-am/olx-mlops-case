from __future__ import annotations

from pathlib import Path

from mlt.model.sklearn import SklearnModel


class Model(SklearnModel):
    """Iris flower classifier using RandomForest.

    Inherits from SklearnModel: only fit() needs to be implemented.
    predict(), save(), and load() are provided by SklearnModel and shown
    below for clarity — override them only if you need custom behaviour.
    """

    def fit(self, data_dir: Path, params: dict) -> dict[str, float]:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        # Load data — data_dir is a read-only Path passed by the tool.
        df = pd.read_csv(data_dir / "iris.csv")
        X = df.drop(columns="species")
        y = df["species"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        # Build and fit the model using params from params.yaml.
        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth"),
            random_state=self.seed,
        ).fit(X_tr, y_tr)

        # Return metrics — only numbers allowed.
        return {"accuracy": float(self.model.score(X_te, y_te))}

    def predict(self, inputs: list[dict]) -> list:
        """Return one prediction per input record.

        inputs is a list of dicts with feature names as keys::

            predict([{"sepal_length": 5.1, "sepal_width": 3.5,
                      "petal_length": 1.4, "petal_width": 0.2}])
            # returns ["setosa"]

        SklearnModel reorders columns to match training order and calls
        self.model.predict(). Override for custom pre/post-processing.
        """
        return super().predict(inputs)

    def save(self, path: Path) -> None:
        """Persist the model to a directory.

        path is a pathlib.Path to a local directory created by mlt.
        SklearnModel serialises self.model to path/model.joblib via joblib.
        Override if your model requires a different serialisation format.
        """
        super().save(path)

    @classmethod
    def load(cls, path: Path) -> "Model":
        """Restore the model from a directory.

        path is a pathlib.Path to the same directory passed to save().
        SklearnModel loads self.model from path/model.joblib via joblib.
        Override if you customised save().
        """
        return super().load(path)
