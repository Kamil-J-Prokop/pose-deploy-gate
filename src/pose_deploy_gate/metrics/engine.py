"""Computation engine for metrics module."""

from pose_deploy_gate.metrics.result import ErrorRateMetrics, LatencyMetrics
from pose_deploy_gate.metrics.statistics import compute_percentiles
from pose_deploy_gate.runner.result import RunResult


class MetricsEngine:
    """Engine for metrics module."""

    @staticmethod
    def _successful_latencies(run_result: RunResult) -> tuple[int, ...]:
        """Get successful latencies from run result."""
        return tuple(
            prediction.timing.elapsed_ns
            for prediction in run_result.predictions
            if prediction.error is None
        )

    @staticmethod
    def _compute_latency(latencies: tuple[int, ...]) -> LatencyMetrics:
        """Compute latency metrics from latencies."""
        if not latencies:
            return LatencyMetrics(
                sample_count=0,
                min_ns=None,
                mean_ns=None,
                p50_ns=None,
                p95_ns=None,
                p99_ns=None,
                max_ns=None,
            )

        p50, p95, p99 = compute_percentiles(latencies)
        return LatencyMetrics(
            sample_count=len(latencies),
            min_ns=min(latencies),
            mean_ns=sum(latencies) / len(latencies),
            p50_ns=p50,
            p95_ns=p95,
            p99_ns=p99,
            max_ns=max(latencies),
        )

    @staticmethod
    def _compute_error_metrics(run_result: RunResult) -> ErrorRateMetrics:
        """Compute prediction error metrics from a run result."""
        total_attempts = len(run_result.predictions)
        failed_predictions = sum(
            prediction.error is not None for prediction in run_result.predictions
        )

        return ErrorRateMetrics(
            total_attempts=total_attempts,
            failed_predictions=failed_predictions,
        )
