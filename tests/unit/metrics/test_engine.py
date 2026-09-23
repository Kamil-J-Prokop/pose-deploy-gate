from pathlib import Path

from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput
from pose_deploy_gate.metrics.engine import MetricsEngine
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
