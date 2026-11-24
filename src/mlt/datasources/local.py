"""Local folder DataSource (MVP implementation)."""

from pathlib import Path

from mlt.datasources.base import DataSource


class LocalDataSource(DataSource):
    """DataSource backed by a local directory."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)

    @property
    def path(self) -> Path:
        return self._data_dir

    def check(self) -> None:
        """Verify the directory exists and is non-empty."""
        from mlt.core.errors import ContractError

        if not self._data_dir.exists():
            raise ContractError(
                f"data directory not found: {self._data_dir}\n"
                f"  hint: create the directory or pass --data <path>"
            )
        files = [f for f in self._data_dir.rglob("*") if f.is_file()]
        if not files:
            raise ContractError(
                f"data directory is empty: {self._data_dir}\n"
                f"  hint: add data files before training"
            )

    def hash_files(self) -> dict:
        from mlt.core.fingerprint import hash_data_dir

        return hash_data_dir(self._data_dir)

    def materialize(self) -> Path:
        # Already local — the runner mounts it read-only.
        return self._data_dir
