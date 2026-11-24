import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlt.model.sklearn import SklearnModel


class Model(SklearnModel):
    """Iris flower classifier using RandomForest.

    Inherits from SklearnModel: only fit needs to be implemented.
    predict, save, and load are provided by SklearnModel.
    """

    def fit(self, data_dir, params):
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
