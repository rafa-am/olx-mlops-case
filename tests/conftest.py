"""Shared pytest fixtures for integration tests."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Skip docker-marked tests when Docker is unavailable
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    from mlt.core.docker_check import docker_available

    if docker_available():
        return  # Docker is running; nothing to skip

    skip_docker = pytest.mark.skip(reason="Docker daemon not available")
    for item in items:
        if item.get_closest_marker("docker"):
            item.add_marker(skip_docker)


# ---------------------------------------------------------------------------
# Iris CSV generator
# ---------------------------------------------------------------------------


def make_iris_csv(data_dir: Path) -> None:
    """Generate iris.csv from sklearn's dataset."""
    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # Rename to snake_case matching the design spec.
    df.columns = [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "species"
    ]
    # Replace numeric species with string labels.
    df["species"] = df["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / "iris.csv", index=False)


# ---------------------------------------------------------------------------
# Iris project fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def iris_template_dir() -> Path:
    """Return the path to the iris template bundled with mlt."""
    import mlt

    return Path(mlt.__file__).parent / "templates" / "iris"


@pytest.fixture()
def iris_project(tmp_path, iris_template_dir) -> dict:
    """Copy the iris template to a fresh tmp dir.

    The template already ships data/iris.csv, so no generation is needed.
    Returns a dict with keys: project_dir, data_dir.
    """
    project_dir = tmp_path / "iris_project"
    shutil.copytree(iris_template_dir, project_dir)
    data_dir = project_dir / "data"
    return {"project_dir": project_dir, "data_dir": data_dir}


# ---------------------------------------------------------------------------
# Minimal project fixture (no sklearn, just a DummyClassifier)
# ---------------------------------------------------------------------------

_MINIMAL_MODEL = """\
    from mlt.model.sklearn import SklearnModel

    class Model(SklearnModel):
        def fit(self, data_dir, params):
            from sklearn.dummy import DummyClassifier
            import pandas as pd
            df = pd.read_csv(data_dir / "data.csv")
            X = df[["x"]]
            y = df["y"]
            self.model = DummyClassifier(strategy="most_frequent")
            self.model.fit(X, y)
            return {"accuracy": float(self.model.score(X, y))}
"""

_MINIMAL_REQ = "scikit-learn>=1.3\npandas>=2.0\n"


@pytest.fixture()
def minimal_project(tmp_path) -> dict:
    """Tiny project with a DummyClassifier and a 10-row CSV."""
    project_dir = tmp_path / "minimal"
    project_dir.mkdir()
    (project_dir / "model.py").write_text(textwrap.dedent(_MINIMAL_MODEL))
    (project_dir / "requirements.txt").write_text(_MINIMAL_REQ)

    data_dir = project_dir / "data"
    data_dir.mkdir()
    # 10-row CSV with a single feature
    csv = "x,y\n" + "\n".join(f"{i},{i%2}" for i in range(10)) + "\n"
    (data_dir / "data.csv").write_text(csv)

    return {"project_dir": project_dir, "data_dir": data_dir}
