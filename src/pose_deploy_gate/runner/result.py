"""Dataclasses for the runner results."""

from dataclasses import dataclass

from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput


@dataclass(frozen=True)
class WarmupResult:
    iterations: int
    total_time_ns: int


@dataclass(frozen=True)
class PredictionTiming:
    image_id: str
    elapsed_ns: int


@dataclass(frozen=True)
class PredictionResult:
    image: ImageInput
    output: AdapterOutput | None
    timing: PredictionTiming
    error: str | None = None


@dataclass(frozen=True)
class RunResult:
    warmup: WarmupResult
    predictions: tuple[PredictionResult, ...]
    total_time_ns: int

    @property
    def successful_predictions(self) -> int:
        return sum(1 for prediction in self.predictions if prediction.error is None)

    @property
    def failed_predictions(self) -> int:
        return sum(1 for prediction in self.predictions if prediction.error is not None)

    @property
    def measured_inference_time_ns(self) -> int:
        return sum(prediction.timing.elapsed_ns for prediction in self.predictions)

    @property
    def average_inference_time_ns(self) -> float:
        successful_predictions = tuple(
            prediction for prediction in self.predictions if prediction.error is None
        )
        if not successful_predictions:
            return 0.0
        return sum(prediction.timing.elapsed_ns for prediction in successful_predictions) / len(
            successful_predictions
        )
