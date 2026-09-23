import pose_deploy_gate.metrics as metrics
from pose_deploy_gate.metrics.engine import MetricsEngine
from pose_deploy_gate.metrics.result import ErrorRateMetrics, LatencyMetrics, MetricsResult


def test_metrics_public_api_exports_expected_symbols() -> None:
    expected_exports = {
        "ErrorRateMetrics": ErrorRateMetrics,
        "LatencyMetrics": LatencyMetrics,
        "MetricsEngine": MetricsEngine,
        "MetricsResult": MetricsResult,
    }

    assert metrics.__all__ == list(expected_exports)
    assert {name: getattr(metrics, name) for name in metrics.__all__} == expected_exports
    assert not hasattr(metrics, "compute_percentiles")
