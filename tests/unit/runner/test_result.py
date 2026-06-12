from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput
from pose_deploy_gate.runner.result import (
    PredictionResult,
    PredictionTiming,
    RunResult,
    WarmupResult,
)


def _image(image_id: str) -> ImageInput:
    return ImageInput(image_id=image_id, path=Path(f"/tmp/{image_id}.jpg"))


def _prediction(
    image_id: str,
    elapsed_ns: int,
    *,
    error: str | None = None,
) -> PredictionResult:
    return PredictionResult(
        image=_image(image_id),
        output=None if error else AdapterOutput(poses=()),
        timing=PredictionTiming(image_id=image_id, elapsed_ns=elapsed_ns),
        error=error,
    )


def test_run_result_counts_successful_predictions() -> None:
    result = RunResult(
        warmup=WarmupResult(iterations=3, total_time_ns=120),
        predictions=(
            _prediction("image-001", 100),
            _prediction("image-002", 200, error="adapter failed"),
            _prediction("image-003", 300),
        ),
        total_time_ns=1_000,
    )

    assert result.successful_predictions == 2


def test_run_result_counts_failed_predictions() -> None:
    result = RunResult(
        warmup=WarmupResult(iterations=3, total_time_ns=120),
        predictions=(
            _prediction("image-001", 100, error="bad input"),
            _prediction("image-002", 200),
            _prediction("image-003", 300, error="timeout"),
        ),
        total_time_ns=1_000,
    )

    assert result.failed_predictions == 2


def test_run_result_sums_measured_inference_time() -> None:
    result = RunResult(
        warmup=WarmupResult(iterations=1, total_time_ns=50),
        predictions=(
            _prediction("image-001", 100),
            _prediction("image-002", 200, error="adapter failed"),
            _prediction("image-003", 300),
        ),
        total_time_ns=1_000,
    )

    assert result.measured_inference_time_ns == 600


def test_run_result_average_inference_time_uses_successful_predictions() -> None:
    result = RunResult(
        warmup=WarmupResult(iterations=1, total_time_ns=50),
        predictions=(
            _prediction("image-001", 100),
            _prediction("image-002", 900, error="adapter failed"),
            _prediction("image-003", 300),
        ),
        total_time_ns=1_000,
    )

    assert result.average_inference_time_ns == 200.0


def test_run_result_average_is_zero_without_successes() -> None:
    result = RunResult(
        warmup=WarmupResult(iterations=1, total_time_ns=50),
        predictions=(
            _prediction("image-001", 100, error="adapter failed"),
            _prediction("image-002", 200, error="timeout"),
        ),
        total_time_ns=1_000,
    )

    assert result.average_inference_time_ns == 0.0


def test_result_types_are_immutable() -> None:
    warmup = WarmupResult(iterations=1, total_time_ns=50)

    with pytest.raises(FrozenInstanceError):
        warmup.iterations = 2
