from pathlib import Path

from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput
from pose_deploy_gate.metrics.engine import MetricsEngine
from pose_deploy_gate.metrics.result import ErrorRateMetrics
from pose_deploy_gate.runner.result import (
    PredictionResult,
    PredictionTiming,
    RunResult,
    WarmupResult,
)


def _prediction(
    image_id: str,
    elapsed_ns: int,
    *,
    error: str | None = None,
) -> PredictionResult:
    return PredictionResult(
        image=ImageInput(image_id=image_id, path=Path(f"/tmp/{image_id}.jpg")),
        output=None if error else AdapterOutput(poses=()),
        timing=PredictionTiming(image_id=image_id, elapsed_ns=elapsed_ns),
        error=error,
    )


def _run_result(*predictions: PredictionResult) -> RunResult:
    return RunResult(
        warmup=WarmupResult(iterations=0, total_time_ns=0),
        predictions=predictions,
        total_time_ns=sum(prediction.timing.elapsed_ns for prediction in predictions),
    )


def _error_metrics(*errors: str | None) -> ErrorRateMetrics:
    run_result = _run_result(
        *(_prediction(f"image-{index}", 100, error=error) for index, error in enumerate(errors))
    )
    return MetricsEngine._compute_error_metrics(run_result)


def test_latency_metrics_use_successful_predictions_only() -> None:
    run_result = _run_result(
        _prediction("success-1", 100),
        _prediction("failure", 10_000, error="adapter failed"),
        _prediction("success-2", 300),
    )

    latencies = MetricsEngine._successful_latencies(run_result)
    metrics = MetricsEngine._compute_latency(latencies)

    assert latencies == (100, 300)
    assert metrics.mean_ns == 200.0


def test_latency_metrics_compute_sample_count() -> None:
    metrics = MetricsEngine._compute_latency((100, 200, 300))

    assert metrics.sample_count == 3


def test_latency_metrics_compute_minimum() -> None:
    metrics = MetricsEngine._compute_latency((300, 100, 200))

    assert metrics.min_ns == 100


def test_latency_metrics_compute_mean() -> None:
    metrics = MetricsEngine._compute_latency((100, 200, 300))

    assert metrics.mean_ns == 200.0


def test_latency_metrics_compute_p50() -> None:
    metrics = MetricsEngine._compute_latency(tuple(range(1, 101)))

    assert metrics.p50_ns == 50.0


def test_latency_metrics_compute_p95() -> None:
    metrics = MetricsEngine._compute_latency(tuple(range(1, 101)))

    assert metrics.p95_ns == 95.0


def test_latency_metrics_compute_p99() -> None:
    metrics = MetricsEngine._compute_latency(tuple(range(1, 101)))

    assert metrics.p99_ns == 99.0


def test_latency_metrics_compute_maximum() -> None:
    metrics = MetricsEngine._compute_latency((300, 100, 200))

    assert metrics.max_ns == 300


def test_latency_metrics_return_none_without_successful_predictions() -> None:
    run_result = _run_result(
        _prediction("failure-1", 100, error="adapter failed"),
        _prediction("failure-2", 300, error="timeout"),
    )

    latencies = MetricsEngine._successful_latencies(run_result)
    metrics = MetricsEngine._compute_latency(latencies)

    assert metrics.sample_count == 0
    assert metrics.min_ns is None
    assert metrics.mean_ns is None
    assert metrics.p50_ns is None
    assert metrics.p95_ns is None
    assert metrics.p99_ns is None
    assert metrics.max_ns is None


def test_error_metrics_count_total_attempts() -> None:
    metrics = _error_metrics(None, "adapter failed", None)

    assert metrics.total_attempts == 3


def test_error_metrics_count_successes() -> None:
    metrics = _error_metrics(None, "adapter failed", None)

    assert metrics.successful_predictions == 2


def test_error_metrics_count_failures() -> None:
    metrics = _error_metrics(None, "adapter failed", "timeout")

    assert metrics.failed_predictions == 2


def test_error_metrics_compute_error_rate() -> None:
    metrics = _error_metrics(None, None, None, "adapter failed")

    assert metrics.error_rate == 0.25


def test_error_metrics_compute_success_rate() -> None:
    metrics = _error_metrics(None, None, None, "adapter failed")

    assert metrics.success_rate == 0.75


def test_error_metrics_report_zero_error_rate_when_all_succeed() -> None:
    metrics = _error_metrics(None, None, None)

    assert metrics.error_rate == 0.0


def test_error_metrics_report_one_error_rate_when_all_fail() -> None:
    metrics = _error_metrics("adapter failed", "timeout", "invalid output")

    assert metrics.error_rate == 1.0


def test_error_metrics_are_undefined_when_no_attempts_exist() -> None:
    metrics = _error_metrics()

    assert metrics.total_attempts == 0
    assert metrics.successful_predictions == 0
    assert metrics.failed_predictions == 0
    assert metrics.error_rate is None
    assert metrics.success_rate is None
