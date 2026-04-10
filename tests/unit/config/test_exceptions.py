from pathlib import Path

import pytest

from pose_deploy_gate.config.exceptions import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
)
from pose_deploy_gate.config.loader import load_config

FIXTURES_DIR = Path("tests/fixtures/config")
INVALID_FIXTURES_DIR = FIXTURES_DIR / "invalid"


def test_config_file_not_found_error_is_a_config_error() -> None:
    assert issubclass(ConfigFileNotFoundError, ConfigError)


def test_config_parse_error_is_a_config_error() -> None:
    assert issubclass(ConfigParseError, ConfigError)


def test_config_validation_error_is_a_config_error() -> None:
    assert issubclass(ConfigValidationError, ConfigError)


def test_loader_raises_config_error_family_for_missing_config() -> None:
    missing_path = FIXTURES_DIR / "does_not_exist.yaml"

    with pytest.raises(ConfigError):
        load_config(missing_path)


def test_loader_raises_config_parse_error_for_malformed_yaml() -> None:
    config_path = INVALID_FIXTURES_DIR / "malformed.yaml"

    with pytest.raises(ConfigParseError):
        load_config(config_path)


def test_loader_raises_config_validation_error_for_schema_invalid_config() -> None:
    config_path = INVALID_FIXTURES_DIR / "invalid_device.yaml"

    with pytest.raises(ConfigValidationError):
        load_config(config_path)
