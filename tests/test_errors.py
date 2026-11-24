"""Tests for exit codes and exception hierarchy."""

from mlt.core.errors import (
    ContractError,
    ExecutionError,
    ExitCode,
    MltEnvironmentError,
    MltError,
    MltTimeoutError,
    NotFoundError,
    OperationRefusedError,
    UsageError,
)


def test_exit_codes_all_present():
    expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 130}
    actual = {code.value for code in ExitCode}
    assert actual == expected


def test_each_error_carries_correct_exit_code():
    assert UsageError("x").exit_code == ExitCode.USAGE_ERROR
    assert ContractError("x").exit_code == ExitCode.CONTRACT_ERROR
    assert ExecutionError("x").exit_code == ExitCode.EXECUTION_ERROR
    assert MltTimeoutError("x").exit_code == ExitCode.TIMEOUT
    assert MltEnvironmentError("x").exit_code == ExitCode.ENVIRONMENT_ERROR
    assert NotFoundError("x").exit_code == ExitCode.NOT_FOUND
    assert OperationRefusedError("x").exit_code == ExitCode.OPERATION_REFUSED


def test_all_errors_subclass_mlt_error():
    for cls in (
        UsageError,
        ContractError,
        ExecutionError,
        MltTimeoutError,
        MltEnvironmentError,
        NotFoundError,
        OperationRefusedError,
    ):
        assert issubclass(cls, MltError), f"{cls.__name__} must extend MltError"


def test_error_message_accessible():
    err = ContractError("bad contract")
    assert str(err) == "bad contract"
    assert err.message == "bad contract"


def test_mlt_error_default_exit_code():
    # MltError itself maps to UNEXPECTED_ERROR (1)
    assert MltError("x").exit_code == ExitCode.UNEXPECTED_ERROR
