"""Exceptions for the runner layer."""


class RunnerError(Exception):
    """Base exception for runner errors."""


class RunnerExecutionError(RunnerError):
    """Exception raised when an error occurs during runner execution."""
