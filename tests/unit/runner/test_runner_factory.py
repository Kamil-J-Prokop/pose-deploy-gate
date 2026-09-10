from pathlib import Path

from pose_deploy_gate.config import AppConfig
from pose_deploy_gate.runner import create_runner
from pose_deploy_gate.runner.runner import Runner


def test_create_runner_returns_runner(tmp_path: Path) -> None:
    config = AppConfig(
        version=1,
        adapter={"type": "dummy", "params": {"pose_confidence": 0.75}},
        data={"input_dir": tmp_path, "file_pattern": "*.jpg", "recursive": True},
        runner={"warmup_iterations": 5, "continue_on_error": True},
    )

    runner = create_runner(config)

    assert isinstance(runner, Runner)


def test_create_runner_uses_adapter_config(tmp_path: Path) -> None:
    config = AppConfig(
        version=1,
        adapter={"type": "dummy", "params": {"pose_confidence": 0.75}},
        data={"input_dir": tmp_path, "file_pattern": "*.jpg", "recursive": True},
        runner={"warmup_iterations": 5, "continue_on_error": True},
    )

    runner = create_runner(config)

    assert runner.adapter.pose_confidence == 0.75


def test_create_runner_uses_data_config(tmp_path: Path) -> None:
    config = AppConfig(
        version=1,
        adapter={"type": "dummy", "params": {"pose_confidence": 0.75}},
        data={"input_dir": tmp_path, "file_pattern": "*.jpg", "recursive": True},
        runner={"warmup_iterations": 5, "continue_on_error": True},
    )

    runner = create_runner(config)

    assert runner.data_source.input_dir == tmp_path
    assert runner.data_source.file_pattern == "*.jpg"
    assert runner.data_source.recursive is True


def test_create_runner_uses_runner_config(tmp_path: Path) -> None:
    config = AppConfig(
        version=1,
        adapter={"type": "dummy", "params": {"pose_confidence": 0.75}},
        data={"input_dir": tmp_path, "file_pattern": "*.jpg", "recursive": True},
        runner={"warmup_iterations": 5, "continue_on_error": True},
    )

    runner = create_runner(config)

    assert runner.warmup_iterations == 5
    assert runner.continue_on_error is True
