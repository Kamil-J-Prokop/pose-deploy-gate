"""Adapter-specific exceptions hierarchy."""


class AdapterError(Exception):
    """Base class for adapter-related errors."""


class UnsupportedAdapterError(AdapterError):
    """Raised when an adapter type is not supported."""


class AdapterExecutionError(AdapterError):
    """Raised when adapter prediction fails."""


__all__ = ["AdapterError", "AdapterExecutionError", "UnsupportedAdapterError"]
