from pose_deploy_gate.runner.exceptions import RunnerError, RunnerExecutionError


def test_runner_execution_error_is_runner_error() -> None:
    error = RunnerExecutionError("runner failed")

    assert isinstance(error, RunnerError)
