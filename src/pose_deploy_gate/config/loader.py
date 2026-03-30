from pathlib import Path

import yaml

from pose_deploy_gate.config.models import AppConfig


def load_config(config_path: Path) -> AppConfig:
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    if raw_config is None:
        raise ValueError(f"Config file {config_path} is empty or invalid.")

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file {config_path} must contain a dictionary at the top level.")

    return AppConfig(**raw_config)
