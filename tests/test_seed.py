"""Tests for seed resolution and injection."""

import pytest

from mlt.core.errors import ContractError
from mlt.core.seed import resolve_seed


def test_default_seed():
    seed, origin = resolve_seed(None, env={})
    assert seed == 42
    assert origin == "default"


def test_seed_from_env():
    seed, origin = resolve_seed(None, env={"MLT_SEED": "7"})
    assert seed == 7
    assert origin == "env"


def test_seed_flag_takes_precedence_over_env():
    seed, origin = resolve_seed(99, env={"MLT_SEED": "7"})
    assert seed == 99
    assert origin == "cli"


def test_invalid_seed_flag_raises():
    with pytest.raises(ContractError):
        resolve_seed(-1, env={})


def test_invalid_seed_env_raises():
    with pytest.raises(ContractError):
        resolve_seed(None, env={"MLT_SEED": "not_a_number"})


def test_zero_seed_is_valid():
    seed, origin = resolve_seed(0, env={})
    assert seed == 0
    assert origin == "cli"


def test_bool_seed_raises():
    # bool is a subclass of int in Python; True == 1 but should not be accepted.
    with pytest.raises(ContractError):
        resolve_seed(True, env={})  # type: ignore[arg-type]
