from __future__ import annotations


class ConfigError(Exception):
    """Base exception for all config loading, parsing, and validation failures."""


class ConfigFileNotFoundError(ConfigError):
    """Raised when the requested config path does not exist."""


class ConfigParseError(ConfigError):
    """Raised when the config file cannot be parsed into valid YAML content."""


class ConfigValidationError(ConfigError):
    """Raised when a parsed config violates application validation rules."""
