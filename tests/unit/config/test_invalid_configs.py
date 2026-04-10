from pathlib import Path

import pytest

from pose_deploy_gate.config.exceptions import ConfigParseError, ConfigValidationError
from pose_deploy_gate.config.loader import load_config

FIXTURES_DIR = Path("tests/fixtures/config/invalid")


def test_load_config_rejects_missing_required_fields_fixture() -> None:
    fixture_path = FIXTURES_DIR / "missing_required_fields.yaml"

    with pytest.raises(ConfigValidationError, match="validation failed|schema"):
        load_config(fixture_path)


def test_load_config_rejects_invalid_device_fixture() -> None:
    fixture_path = FIXTURES_DIR / "invalid_device.yaml"

    with pytest.raises(ConfigValidationError, match="validation failed|schema"):
        load_config(fixture_path)


def test_load_config_rejects_unknown_key_fixture() -> None:
    fixture_path = FIXTURES_DIR / "unknown_key.yaml"

    with pytest.raises(ConfigValidationError, match="validation failed|schema"):
        load_config(fixture_path)


def test_load_config_rejects_empty_file_fixture() -> None:
    fixture_path = FIXTURES_DIR / "empty.yaml"

    with pytest.raises(ConfigParseError, match="empty"):
        load_config(fixture_path)


def test_load_config_rejects_malformed_yaml_fixture() -> None:
    fixture_path = FIXTURES_DIR / "malformed.yaml"

    with pytest.raises(ConfigParseError, match="Failed to parse YAML config"):
        load_config(fixture_path)
