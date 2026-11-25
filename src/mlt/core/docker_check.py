"""Docker availability check."""

from __future__ import annotations

import subprocess


def docker_available() -> bool:
    """Return True if Docker is installed and the daemon is responding."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
