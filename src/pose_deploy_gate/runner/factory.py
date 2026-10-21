from pose_deploy_gate.adapters import create_adapter
from pose_deploy_gate.config import AppConfig
from pose_deploy_gate.data import create_data_source
from pose_deploy_gate.runner.exceptions import RunnerError
from pose_deploy_gate.runner.runner import Runner


def create_runner(config: AppConfig) -> Runner:
    if len(config.models) != 1:
        raise RunnerError(
            "Execution currently requires exactly one configured model; multi-model execution is not implemented yet."
        )
    adapter = create_adapter(config.models[0].adapter)
    data_source = create_data_source(config.data)
    return Runner(
        adapter=adapter,
        data_source=data_source,
        warmup_iterations=config.runner.warmup_iterations,
        continue_on_error=config.runner.continue_on_error,
    )
