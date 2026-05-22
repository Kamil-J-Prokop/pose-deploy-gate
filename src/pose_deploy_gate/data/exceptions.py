"""Placeholder for data-layer exceptions."""


class DataSourceError(Exception):
    """Base class for data source errors."""


class InputDirectoryError(DataSourceError):
    """Raised when the configured input directory is invalid."""


class NoInputFilesError(DataSourceError):
    """Raised when no input files match the data source configuration."""
