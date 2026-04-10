from pathlib import Path

import pytest
import yaml

from pose_deploy_gate.config.exceptions import (
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
)
from pose_deploy_gate.config.models import AppConfig
from pose_deploy_gate.config.validators import (
    validate_app_config,
    validate_config_path,
    validate_raw_config,
)

FIXTURES_DIR = Path("tests/fixtures/config")


def test_validate_config_path_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing_config.yaml"

    with pytest.raises(ConfigFileNotFoundError, match="does not exist"):
        validate_config_path(missing_path)


def test_validate_raw_config_raises_for_empty_yaml() -> None:
    config_path = FIXTURES_DIR / "empty.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    with pytest.raises(ConfigParseError, match="is empty"):
        validate_raw_config(raw_config, config_path)


def test_validate_raw_config_raises_for_non_mapping_yaml() -> None:
    config_path = FIXTURES_DIR / "top_level_list.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    with pytest.raises(ConfigParseError, match="top-level mapping"):
        validate_raw_config(raw_config, config_path)


def test_validate_app_config_raises_for_missing_input_dir() -> None:
    config_path = FIXTURES_DIR / "missing_input_dir.yaml"
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = AppConfig(**raw_config)

    with pytest.raises(ConfigValidationError, match="input_dir does not exist"):
        validate_app_config(config, config_path)


def test_validation_accepts_valid_config(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    config_path = tmp_path / "valid.yaml"
    config_path.write_text(
        f"""
version: 1

data:
  input_dir: "{input_dir}"

adapter:
  type: "dummy"
""".strip(),
        encoding="utf-8",
    )

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = AppConfig(**raw_config)

    validate_config_path(config_path)
    validate_raw_config(raw_config, config_path)
    validate_app_config(config, config_path)

    assert config.version == 1
    assert config.data.input_dir == input_dir
    assert config.adapter.type == "dummy"
    assert config.run.name == "default-run"


def test_validate_config_path_raises_for_invalid_extension(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match=".yaml or .yml"):
        validate_config_path(config_path)
