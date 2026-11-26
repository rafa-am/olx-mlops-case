"""DataSource interface.

MVP: local folder. Evolution: S3, lakehouse.
The DataSource abstracts where data lives so the runner contract stays
stable when the storage backend changes.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class DataSource(ABC):
    @abstractmethod
    def check(self) -> None:
        """Verify the data source is accessible and usable.

        Raises ContractError (exit 3) on failure.
        Called by ``mlt check`` and before training.
        """

    @abstractmethod
    def hash_files(self) -> dict:
        """Return a dict with per-file hashes and an aggregate hash.

        Schema: {"files": {relative_path: sha256}, "hash": aggregate_sha256}
        """

    @abstractmethod
    def materialize(self) -> Path:
        """Return a local read-only path to the data directory.

        For a local source this is a no-op.
        For a remote source this triggers download/mount.
        """
