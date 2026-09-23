from dataclasses import FrozenInstanceError

import pytest

from pose_deploy_gate.metrics.result import ErrorRateMetrics, LatencyMetrics, MetricsResult


def test_latency_metrics_store_expected_values() -> None:
    lat_metrics = LatencyMetrics(
        sample_count=5,
        min_ns=100.0,
        max_ns=500.0,
        mean_ns=300.0,
        p50_ns=250.0,
        p95_ns=450.0,
        p99_ns=490.0,
    )

    assert lat_metrics.sample_count == 5
    assert lat_metrics.min_ns == 100.0
    assert lat_metrics.max_ns == 500.0
    assert lat_metrics.mean_ns == 300.0
    assert lat_metrics.p50_ns == 250.0
    assert lat_metrics.p95_ns == 450.0
    assert lat_metrics.p99_ns == 490.0


def test_error_rate_metrics_stores_expected_values() -> None:
    error_metrics = ErrorRateMetrics(total_attempts=10, failed_predictions=2)

    assert error_metrics.total_attempts == 10
    assert error_metrics.failed_predictions == 2


def test_error_rate_metrics_derives_values_from_attempt_counts() -> None:
    metrics = ErrorRateMetrics(total_attempts=4, failed_predictions=1)

    assert metrics.successful_predictions == 3
    assert metrics.success_rate == 0.75
    assert metrics.error_rate == 0.25


def test_metrics_result_groups_latency_and_error_metrics() -> None:
    latency_metrics = LatencyMetrics(
        sample_count=5,
        min_ns=100.0,
        max_ns=500.0,
        mean_ns=300.0,
        p50_ns=250.0,
        p95_ns=450.0,
        p99_ns=490.0,
    )
    error_metrics = ErrorRateMetrics(total_attempts=5, failed_predictions=1)

    metrics_result = MetricsResult(latency=latency_metrics, errors=error_metrics)

    assert metrics_result.latency == latency_metrics
    assert metrics_result.errors == error_metrics


@pytest.mark.parametrize(
    ("total_attempts", "failed_predictions"),
    [
        (-1, 0),
        (5, 6),
        (5, -1),
    ],
)
def test_error_rate_metrics_raises_value_error_for_invalid_inputs(
    total_attempts: int,
    failed_predictions: int,
) -> None:
    with pytest.raises(ValueError):
        ErrorRateMetrics(
            total_attempts=total_attempts,
            failed_predictions=failed_predictions,
        )


def test_metric_results_are_immutable() -> None:
    latency_metrics = LatencyMetrics(
        sample_count=5,
        min_ns=100.0,
        max_ns=500.0,
        mean_ns=300.0,
        p50_ns=250.0,
        p95_ns=450.0,
        p99_ns=490.0,
    )
    error_metrics = ErrorRateMetrics(total_attempts=5, failed_predictions=1)
    metrics_result = MetricsResult(latency=latency_metrics, errors=error_metrics)

    with pytest.raises(FrozenInstanceError):
        latency_metrics.sample_count = 10

    with pytest.raises(FrozenInstanceError):
        error_metrics.total_attempts = 10

    with pytest.raises(FrozenInstanceError):
        metrics_result.latency = latency_metrics


def test_latency_metrics_allow_undefined_values() -> None:
    metrics = LatencyMetrics(
        sample_count=0,
        min_ns=None,
        max_ns=None,
        mean_ns=None,
        p50_ns=None,
        p95_ns=None,
        p99_ns=None,
    )

    assert metrics.sample_count == 0
    assert metrics.min_ns is None
    assert metrics.max_ns is None
    assert metrics.mean_ns is None
    assert metrics.p50_ns is None
    assert metrics.p95_ns is None
    assert metrics.p99_ns is None


def test_error_rates_allow_undefined_values() -> None:
    metrics = ErrorRateMetrics(total_attempts=0, failed_predictions=0)

    assert metrics.successful_predictions == 0
    assert metrics.success_rate is None
    assert metrics.error_rate is None


def test_error_rate_metrics_successful_predictions_with_all_failures() -> None:
    metrics = ErrorRateMetrics(total_attempts=5, failed_predictions=5)

    assert metrics.successful_predictions == 0
    assert metrics.success_rate == 0.0
    assert metrics.error_rate == 1.0


def test_error_rate_metrics_successful_predictions_with_all_successes() -> None:
    metrics = ErrorRateMetrics(total_attempts=5, failed_predictions=0)

    assert metrics.successful_predictions == 5
    assert metrics.success_rate == 1.0
    assert metrics.error_rate == 0.0


def test_error_rate_metrics_successful_predictions_with_one_attempt() -> None:
    metrics = ErrorRateMetrics(total_attempts=1, failed_predictions=0)

    assert metrics.successful_predictions == 1
    assert metrics.success_rate == 1.0
    assert metrics.error_rate == 0.0

    metrics = ErrorRateMetrics(total_attempts=1, failed_predictions=1)

    assert metrics.successful_predictions == 0
    assert metrics.success_rate == 0.0
    assert metrics.error_rate == 1.0


def test_error_rate_metrics_successful_predictions_with_large_numbers() -> None:
    metrics = ErrorRateMetrics(total_attempts=1_000_000, failed_predictions=250_000)

    assert metrics.successful_predictions == 750_000
    assert metrics.success_rate == 0.75
    assert metrics.error_rate == 0.25
