from pose_deploy_gate.adapters import create_adapter
from pose_deploy_gate.config.models import DataConfig, ModelConfig, RunnerConfig
from pose_deploy_gate.data import FileDataSource, create_data_source
from pose_deploy_gate.runner.runner import Runner


def create_model_runner(
    model: ModelConfig,
    *,
    data_config: DataConfig,
    runner_config: RunnerConfig,
    data_source: FileDataSource | None = None,
) -> Runner:
    """Create a model runner, optionally reusing a session's data source.

    When supplied, ``data_source`` takes precedence over ``data_config``.
    """
    adapter = create_adapter(model.adapter)
    if data_source is None:
        data_source = create_data_source(data_config)
    return Runner(
        adapter=adapter,
        data_source=data_source,
        warmup_iterations=runner_config.warmup_iterations,
        continue_on_error=runner_config.continue_on_error,
    )
