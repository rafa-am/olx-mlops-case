"""Core model contract.

MLModel is the only abstraction the tool depends on.
No external imports — stdlib only — so this module is importable
anywhere without installing scientist dependencies.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class MLModel(ABC):
    """Abstract base for all mlt models.

    The tool assigns ``self.seed`` before calling ``fit``.
    Subclasses must implement all four methods.

    For scikit-learn models, inherit from ``mlt.model.sklearn.SklearnModel``
    instead — it provides ``predict``, ``save``, and ``load``; only ``fit``
    needs to be written.
    """

    # Assigned by the tool before fit; use it in estimators for reproducibility.
    seed: int

    @abstractmethod
    def fit(self, data_dir: Path, params: dict) -> dict[str, float]:
        """Train the model and return metrics as {name: number}.

        ``data_dir`` is a read-only local directory.
        ``params`` comes from ``params.yaml`` (empty dict if absent).
        The trained model must stay in ``self`` for ``save`` to work.
        Everything printed here goes to ``train.log``.
        """

    @abstractmethod
    def predict(self, inputs: list[dict]) -> list:
        """Return one prediction per record, in the same order.

        ``inputs`` is a list of dicts with feature names as keys.
        Output is free-form but must be JSON-serializable.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artifacts to ``path`` (a local directory created by the tool).

        Write any files needed for ``load`` to reconstruct the model.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "MLModel":
        """Reconstruct the model from ``path`` (the directory written by ``save``)."""
