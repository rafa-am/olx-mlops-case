"""Manifest: the source of truth for a run.

A Manifest wraps the JSON dict defined in docs/design.md §9.
Typed properties expose the most-accessed fields; the raw dict is
always available via ``to_dict()`` and ``Manifest.from_dict()``.

Write is atomic: the file is written to a ``.tmp`` sibling and then
renamed, so a crash during write never leaves a corrupt manifest.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Manifest:
    """Wrapper around the manifest JSON dict."""

    MANIFEST_VERSION = 1

    def __init__(self, data: dict[str, Any]) -> None:
        self._d = data

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._d["run_id"]

    @property
    def status(self) -> str:
        return self._d["status"]

    @property
    def created_at(self) -> str:
        return self._d["created_at"]

    @property
    def finished_at(self) -> str | None:
        return self._d.get("finished_at")

    @property
    def duration_s(self) -> float | None:
        return self._d.get("duration_s")

    @property
    def fingerprint(self) -> str | None:
        return self._d.get("fingerprint")

    @property
    def error(self) -> str | None:
        return self._d.get("error")

    @property
    def verification_of(self) -> str | None:
        return self._d.get("verification_of")

    # ------------------------------------------------------------------
    # Inputs — shortcuts used by RunStore indexing and fingerprint checks
    # ------------------------------------------------------------------

    @property
    def inputs(self) -> dict:
        return self._d.get("inputs", {})

    @property
    def inputs_code_hash(self) -> str | None:
        return self.inputs.get("code", {}).get("hash")

    @property
    def inputs_params_hash(self) -> str | None:
        return self.inputs.get("params", {}).get("hash")

    @property
    def inputs_deps_hash(self) -> str | None:
        return self.inputs.get("deps", {}).get("hash")

    @property
    def inputs_python_requested(self) -> str | None:
        return self.inputs.get("python", {}).get("requested")

    @property
    def inputs_seed(self) -> int | None:
        return self.inputs.get("seed")

    @property
    def inputs_seed_source(self) -> str | None:
        return self.inputs.get("seed_source")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    @property
    def environment(self) -> dict:
        return self._d.get("environment", {})

    @property
    def env_runner(self) -> str | None:
        return self.environment.get("runner")

    @property
    def env_isolation(self) -> str | None:
        return self.environment.get("isolation")

    @property
    def env_python(self) -> str | None:
        return self.environment.get("python")

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    @property
    def outputs(self) -> dict | None:
        return self._d.get("outputs")

    @property
    def metrics(self) -> dict[str, float] | None:
        if self.outputs:
            return self.outputs.get("metrics")
        return None

    # ------------------------------------------------------------------
    # Mutation helpers (called by RunStore)
    # ------------------------------------------------------------------

    def mark_succeeded(
        self,
        metrics: dict[str, float],
        schema_outputs: dict,
        artifacts: dict[str, str],
        deps_lock: list[str],
        env_hash: str,
    ) -> None:
        """Fill in the final outputs and mark the run as succeeded."""
        now = datetime.now(timezone.utc)
        created = datetime.fromisoformat(self._d["created_at"])
        self._d["status"] = "succeeded"
        self._d["finished_at"] = now.isoformat()
        self._d["duration_s"] = round((now - created).total_seconds(), 3)
        self._d["outputs"] = {
            "metrics": metrics,
            "schema": schema_outputs,
            "artifacts": artifacts,
        }
        # Fill in the deps lock (available only after build)
        if "inputs" in self._d and "deps" in self._d["inputs"]:
            self._d["inputs"]["deps"]["lock"] = deps_lock
        if "environment" in self._d:
            self._d["environment"]["env_hash"] = env_hash

    def mark_failed(self, error: str) -> None:
        """Fill in the error and mark the run as failed."""
        now = datetime.now(timezone.utc)
        created = datetime.fromisoformat(self._d["created_at"])
        self._d["status"] = "failed"
        self._d["finished_at"] = now.isoformat()
        self._d["duration_s"] = round((now - created).total_seconds(), 3)
        self._d["error"] = error

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return self._d

    @classmethod
    def from_dict(cls, data: dict) -> Manifest:
        return cls(data)

    def write(self, path: Path) -> None:
        """Write atomically: write to .tmp then rename."""
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._d, indent=2), encoding="utf-8")
        tmp.rename(path)

    @classmethod
    def read(cls, path: Path) -> Manifest:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def new(
        cls,
        run_id: str,
        inputs: dict,
        fingerprint: str,
        environment: dict,
    ) -> Manifest:
        """Build a partial manifest (status: running) for a new run."""
        return cls(
            {
                "manifest_version": cls.MANIFEST_VERSION,
                "run_id": run_id,
                "status": "running",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "duration_s": None,
                "inputs": inputs,
                "fingerprint": fingerprint,
                "environment": environment,
                "outputs": None,
                "error": None,
                "verification_of": None,
            }
        )
