from __future__ import annotations

from pathlib import Path

from pose_deploy_gate.config.models import AppConfig


def validate_config_path(config_path: Path) -> None:
    """Validate config file path concerns before YAML parsing."""
    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")

    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Config file must have a .yaml or .yml extension: {config_path}")


def validate_raw_config(raw_config: object, config_path: Path) -> None:
    """Validate the parsed YAML payload before model construction."""
    if raw_config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {config_path}")


def validate_app_config(config: AppConfig, config_path: Path) -> None:
    """Validate AppConfig rules that depend on resolved model values."""
    if not config.data.input_dir.exists():
        raise ValueError(
            f"Configured input_dir does not exist: {config.data.input_dir} (from {config_path})"
        )

    if not config.data.input_dir.is_dir():
        raise ValueError(
            f"Configured input_dir is not a directory: {config.data.input_dir} (from {config_path})"
        )

    if config.run.name.strip() == "":
        raise ValueError(f"Configured run.name cannot be empty or whitespace (from {config_path})")

    if config.data.file_pattern.strip() == "":
        raise ValueError(
            f"Configured data.file_pattern cannot be empty or whitespace (from {config_path})"
        )

    if config.run.seed is not None and config.run.seed < 0:
        raise ValueError(f"Configured run.seed must be non-negative (from {config_path})")
