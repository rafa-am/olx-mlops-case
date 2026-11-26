"""Seed resolution and injection.

Precedence: --seed flag > MLT_SEED env var > 42 (default).
Applies seed to random, numpy (if available), and PYTHONHASHSEED.
"""

from __future__ import annotations

from typing import Mapping


def resolve_seed(
    seed_flag: int | None,
    env: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    """Return (seed_value, origin) where origin is 'cli', 'env', or 'default'.

    Raises ContractError (exit 3) if the value is not a non-negative integer.
    """
    import os

    if env is None:
        env = os.environ

    if seed_flag is not None:
        _validate(seed_flag, source="--seed flag")
        return int(seed_flag), "cli"

    env_val = env.get("MLT_SEED")
    if env_val is not None:
        try:
            seed = int(env_val)
        except (ValueError, TypeError):
            from mlt.core.errors import ContractError

            raise ContractError(
                f"MLT_SEED={env_val!r} is not a valid integer; "
                "set a non-negative integer or unset the variable"
            )
        _validate(seed, source="MLT_SEED")
        return seed, "env"

    return 42, "default"


def apply_seed(seed: int) -> None:
    """Fix random, numpy (if importable) and PYTHONHASHSEED to seed."""
    import os
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _validate(value: int, source: str) -> None:
    from mlt.core.errors import ContractError

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(
            f"seed from {source} must be a non-negative integer, got {value!r}"
        )
