from pathlib import Path

import pytest
from pydantic import ValidationError

from pose_deploy_gate.config import AppConfig
from pose_deploy_gate.config.models import RunnerConfig


def test_minimal_valid_config():
    config = AppConfig(version=1, data={"input_dir": "./data"}, adapter={"type": "dummy"})
    assert config.version == 1
    assert config.data.input_dir == Path("data")
    assert config.run.name == "default-run"
    assert config.adapter.type == "dummy"
    assert config.output.dir == Path("./artifacts")
    assert not config.gates.enabled
    assert config.runner == RunnerConfig()


def test_runner_config_uses_defaults() -> None:
    config = RunnerConfig()

    assert config.warmup_iterations == 3
    assert config.continue_on_error is False


def test_runner_config_accepts_zero_warmup_iterations() -> None:
    config = RunnerConfig(warmup_iterations=0)

    assert config.warmup_iterations == 0


def test_runner_config_rejects_negative_warmup_iterations() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        RunnerConfig(warmup_iterations=-1)


def test_app_config_uses_default_runner_config() -> None:
    config = AppConfig(version=1, data={"input_dir": "./data"}, adapter={"type": "dummy"})

    assert config.runner == RunnerConfig()


def test_app_config_accepts_runner_config() -> None:
    config = AppConfig(
        version=1,
        data={"input_dir": "./data"},
        adapter={"type": "dummy"},
        runner={"warmup_iterations": 0, "continue_on_error": True},
    )

    assert config.runner == RunnerConfig(warmup_iterations=0, continue_on_error=True)
