from pathlib import Path

import yaml
from pydantic import ValidationError

from pose_deploy_gate.config.exceptions import ConfigParseError, ConfigValidationError
from pose_deploy_gate.config.models import AppConfig
from pose_deploy_gate.config.validators import (
    validate_app_config,
    validate_config_path,
    validate_raw_config,
)


def load_config(config_path: str | Path) -> AppConfig:
    """Load the YAML config and return a validated AppConfig instance."""
    config_path = Path(config_path)

    validate_config_path(config_path)

    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw_config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigParseError(f"Failed to parse YAML config: {config_path}") from exc

    validate_raw_config(raw_config, config_path)

    try:
        config = AppConfig(**raw_config)
    except ValidationError as exc:
        raise ConfigValidationError(f"Config schema validation failed: {config_path}") from exc

    validate_app_config(config, config_path)
    return config
