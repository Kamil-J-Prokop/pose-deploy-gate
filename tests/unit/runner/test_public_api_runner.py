from pose_deploy_gate.runner import (
    PredictionFailureKind,
    PredictionResult,
    PredictionTiming,
    Runner,
    RunnerError,
    RunnerExecutionError,
    RunResult,
    Timer,
    WarmupResult,
    create_runner,
    ns_to_ms,
)


def test_runner_public_api_imports_expected_symbols() -> None:
    assert PredictionFailureKind is not None
    assert PredictionResult is not None
    assert PredictionTiming is not None
    assert RunResult is not None
    assert Runner is not None
    assert RunnerError is not None
    assert RunnerExecutionError is not None
    assert Timer is not None
    assert WarmupResult is not None
    assert create_runner is not None
    assert ns_to_ms is not None
