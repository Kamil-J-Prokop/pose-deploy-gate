from pathlib import Path

import pytest

from pose_deploy_gate.config.models import DataConfig, ModelConfig, RunnerConfig
from pose_deploy_gate.data import FileDataSource
from pose_deploy_gate.runner import Runner, create_model_runner


@pytest.mark.parametrize("warmup_iterations,continue_on_error", [(0, False), (5, True)])
def test_create_model_runners_use_requested_models_and_shared_settings(
    tmp_path: Path, warmup_iterations: int, continue_on_error: bool
) -> None:
    models = [
        ModelConfig(
            id=model_id,
            adapter={"type": "dummy", "params": {"pose_confidence": confidence}},
        )
        for model_id, confidence in [("reference", 0.9), ("candidate", 0.6)]
    ]
    data_config = DataConfig(input_dir=tmp_path, file_pattern="*.jpg", recursive=True)
    runner_config = RunnerConfig(
        warmup_iterations=warmup_iterations, continue_on_error=continue_on_error
    )
    runners = [
        create_model_runner(model, data_config=data_config, runner_config=runner_config)
        for model in models
    ]

    assert runners[0].adapter is not runners[1].adapter
    assert runners[0].adapter.pose_confidence == 0.9
    assert runners[1].adapter.pose_confidence == 0.6
    for runner in runners:
        assert isinstance(runner, Runner)
        assert runner.data_source.input_dir == data_config.input_dir
        assert runner.data_source.file_pattern == data_config.file_pattern
        assert runner.data_source.recursive == data_config.recursive
        assert runner.warmup_iterations == warmup_iterations
        assert runner.continue_on_error == continue_on_error


def test_repeated_creation_for_same_model_uses_separate_adapters(tmp_path: Path) -> None:
    model = ModelConfig(id="model", adapter={"type": "dummy"})
    data_config = DataConfig(input_dir=tmp_path)
    runner_config = RunnerConfig()
    first = create_model_runner(model, data_config=data_config, runner_config=runner_config)
    second = create_model_runner(model, data_config=data_config, runner_config=runner_config)
    assert first.adapter is not second.adapter
    assert first.warmup_iterations == second.warmup_iterations == 3
    assert first.continue_on_error is second.continue_on_error is False


def test_create_model_runner_reuses_supplied_data_source(tmp_path: Path) -> None:
    source = FileDataSource(tmp_path, "*.jpg", recursive=True)
    runner = create_model_runner(
        ModelConfig(id="model", adapter={"type": "dummy"}),
        data_config=DataConfig(input_dir=tmp_path / "unused"),
        runner_config=RunnerConfig(warmup_iterations=0, continue_on_error=True),
        data_source=source,
    )

    assert runner.data_source is source
    assert runner.warmup_iterations == 0
    assert runner.continue_on_error is True
