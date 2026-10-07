"""Core runner orchestration logic for the pose deploy gate."""

from dataclasses import dataclass, field

from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.adapters.exceptions import AdapterExecutionError
from pose_deploy_gate.data.datasource import FileDataSource
from pose_deploy_gate.runner.exceptions import RunnerExecutionError
from pose_deploy_gate.runner.result import (
    PredictionFailureKind,
    PredictionResult,
    PredictionTiming,
    RunResult,
    WarmupResult,
)
from pose_deploy_gate.runner.timing import Timer
from pose_deploy_gate.validation import (
    AdapterOutputValidationError,
    AdapterOutputValidator,
)


@dataclass(frozen=True)
class Runner:
    adapter: PoseAdapter
    data_source: FileDataSource
    warmup_iterations: int = 3
    continue_on_error: bool = False
    timer: Timer = field(default_factory=Timer)
    validator: AdapterOutputValidator = field(default_factory=AdapterOutputValidator)

    def run(self) -> RunResult:
        images = tuple(self.data_source.iter_images())
        run_start_ns = self.timer.now_ns()
        run_end_ns = run_start_ns
        actual_warmup_iterations = self.warmup_iterations if images else 0

        warmup_total_ns = 0
        if images and actual_warmup_iterations > 0:
            warmup_start_ns = self.timer.now_ns()
            try:
                for _ in range(actual_warmup_iterations):
                    output = self.adapter.predict(images[0])
                    self.validator.validate_or_raise(output)
            except AdapterExecutionError as e:
                raise RunnerExecutionError(f"Adapter warmup failed: {e}") from e
            except AdapterOutputValidationError as e:
                raise RunnerExecutionError(f"Adapter warmup output validation failed: {e}") from e
            warmup_total_ns = self.timer.elapsed_ns(warmup_start_ns)
            run_end_ns = warmup_start_ns + warmup_total_ns

        predictions: list[PredictionResult] = []
        for image in images:
            prediction_start_ns = self.timer.now_ns()

            try:
                output = self.adapter.predict(image)
                elapsed_ns = self.timer.elapsed_ns(prediction_start_ns)
            except AdapterExecutionError as e:
                elapsed_ns = self.timer.elapsed_ns(prediction_start_ns)
                run_end_ns = prediction_start_ns + elapsed_ns
                predictions.append(
                    PredictionResult(
                        image=image,
                        output=None,
                        timing=PredictionTiming(image_id=image.image_id, elapsed_ns=elapsed_ns),
                        error=str(e),
                        failure_kind=PredictionFailureKind.ADAPTER_EXECUTION,
                    )
                )
                if not self.continue_on_error:
                    raise RunnerExecutionError(
                        f"Adapter execution failed for image {image.image_id}: {e}"
                    ) from e

                continue

            run_end_ns = prediction_start_ns + elapsed_ns

            try:
                self.validator.validate_or_raise(output)
            except AdapterOutputValidationError as e:
                predictions.append(
                    PredictionResult(
                        image=image,
                        output=None,
                        timing=PredictionTiming(image_id=image.image_id, elapsed_ns=elapsed_ns),
                        error=str(e),
                        failure_kind=PredictionFailureKind.OUTPUT_VALIDATION,
                    )
                )
                if not self.continue_on_error:
                    raise RunnerExecutionError(
                        f"Output validation failed for image {image.image_id}: {e}"
                    ) from e
                continue

            predictions.append(
                PredictionResult(
                    image=image,
                    output=output,
                    timing=PredictionTiming(image_id=image.image_id, elapsed_ns=elapsed_ns),
                )
            )

        total_time_ns = run_end_ns - run_start_ns
        return RunResult(
            warmup=WarmupResult(
                iterations=actual_warmup_iterations,
                total_time_ns=warmup_total_ns,
            ),
            predictions=tuple(predictions),
            total_time_ns=total_time_ns,
        )
