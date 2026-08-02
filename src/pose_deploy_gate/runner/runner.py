"""Core runner orchestration logic for the pose deploy gate."""

from dataclasses import dataclass, field

from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.data.datasource import FileDataSource
from pose_deploy_gate.runner.result import (
    PredictionResult,
    PredictionTiming,
    RunResult,
    WarmupResult,
)
from pose_deploy_gate.runner.timing import Timer


@dataclass(frozen=True)
class Runner:
    adapter: PoseAdapter
    data_source: FileDataSource
    warmup_iterations: int = 3
    continue_on_error: bool = False
    timer: Timer = field(default_factory=Timer)

    def run(self) -> RunResult:
        images = tuple(self.data_source.iter_images())
        run_start_ns = self.timer.now_ns()
        run_end_ns = run_start_ns

        warmup_total_ns = 0
        if images and self.warmup_iterations > 0:
            warmup_start_ns = self.timer.now_ns()
            for _ in range(self.warmup_iterations):
                self.adapter.predict(images[0])
            warmup_total_ns = self.timer.elapsed_ns(warmup_start_ns)
            run_end_ns = warmup_start_ns + warmup_total_ns

        predictions: list[PredictionResult] = []
        for image in images:
            prediction_start_ns = self.timer.now_ns()
            output = self.adapter.predict(image)
            elapsed_ns = self.timer.elapsed_ns(prediction_start_ns)
            run_end_ns = prediction_start_ns + elapsed_ns
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
                iterations=self.warmup_iterations,
                total_time_ns=warmup_total_ns,
            ),
            predictions=tuple(predictions),
            total_time_ns=total_time_ns,
        )
