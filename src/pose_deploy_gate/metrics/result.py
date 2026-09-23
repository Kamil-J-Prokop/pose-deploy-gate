"""Dataclasses for metrics results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyMetrics:
    """Latency metrics dataclass; values are None if there are no successful predictions."""

    sample_count: int
    min_ns: float | None
    max_ns: float | None
    mean_ns: float | None
    p50_ns: float | None
    p95_ns: float | None
    p99_ns: float | None


@dataclass(frozen=True)
class ErrorRateMetrics:
    total_attempts: int
    failed_predictions: int

    def __post_init__(self) -> None:
        if self.total_attempts < 0:
            raise ValueError("total_attempts cannot be negative")
        if not 0 <= self.failed_predictions <= self.total_attempts:
            raise ValueError("failed_predictions must be between 0 and total_attempts")

    @property
    def successful_predictions(self) -> int:
        return self.total_attempts - self.failed_predictions

    @property
    def success_rate(self) -> float | None:
        if self.total_attempts == 0:
            return None
        return self.successful_predictions / self.total_attempts

    @property
    def error_rate(self) -> float | None:
        if self.total_attempts == 0:
            return None
        return self.failed_predictions / self.total_attempts


@dataclass(frozen=True)
class MetricsResult:
    """Result class for metrics."""

    latency: LatencyMetrics
    errors: ErrorRateMetrics
