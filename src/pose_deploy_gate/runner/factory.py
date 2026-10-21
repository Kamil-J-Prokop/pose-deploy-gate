from pose_deploy_gate.adapters import create_adapter
from pose_deploy_gate.config.models import DataConfig, ModelConfig, RunnerConfig
from pose_deploy_gate.data import create_data_source
from pose_deploy_gate.runner.runner import Runner


def create_model_runner(
    model: ModelConfig,
    *,
    data_config: DataConfig,
    runner_config: RunnerConfig,
) -> Runner:
    """Create a runner with its own adapter for the requested model."""
    adapter = create_adapter(model.adapter)
    data_source = create_data_source(data_config)
    return Runner(
        adapter=adapter,
        data_source=data_source,
        warmup_iterations=runner_config.warmup_iterations,
        continue_on_error=runner_config.continue_on_error,
    )
