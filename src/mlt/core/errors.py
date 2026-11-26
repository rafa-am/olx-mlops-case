"""Exit codes and exception hierarchy for mlt.

The CLI translates exceptions to exit codes; all logic raises these.
Errors go to stderr; results go to stdout.
"""

from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    UNEXPECTED_ERROR = 1
    USAGE_ERROR = 2
    CONTRACT_ERROR = 3
    EXECUTION_ERROR = 4
    TIMEOUT = 5
    ENVIRONMENT_ERROR = 6
    NOT_FOUND = 7
    OPERATION_REFUSED = 8
    INTERRUPTED = 130


class MltError(Exception):
    """Base class for all mlt errors. Carries the exit code the CLI should use."""

    exit_code: ExitCode = ExitCode.UNEXPECTED_ERROR

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UsageError(MltError):
    """Incorrect CLI usage (exit 2)."""

    exit_code = ExitCode.USAGE_ERROR


class ContractError(MltError):
    """Contract violation detected before training; no run is created (exit 3)."""

    exit_code = ExitCode.CONTRACT_ERROR


class ExecutionError(MltError):
    """Runtime failure during training or serving; a failed run is created (exit 4)."""

    exit_code = ExitCode.EXECUTION_ERROR


class MltTimeoutError(MltError):
    """Training or serve --detach timed out (exit 5)."""

    exit_code = ExitCode.TIMEOUT


class MltEnvironmentError(MltError):
    """Environment unavailable: Docker absent, build failed, port busy (exit 6)."""

    exit_code = ExitCode.ENVIRONMENT_ERROR


class NotFoundError(MltError):
    """Project, run, or service not found (exit 7)."""

    exit_code = ExitCode.NOT_FOUND


class OperationRefusedError(MltError):
    """Operation refused due to current state, e.g. serving a failed run (exit 8)."""

    exit_code = ExitCode.OPERATION_REFUSED
