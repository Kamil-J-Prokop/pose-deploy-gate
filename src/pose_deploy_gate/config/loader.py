from pathlib import Path

import yaml

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

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    validate_raw_config(raw_config, config_path)

    config = AppConfig(**raw_config)
    validate_app_config(config, config_path)
    return config
