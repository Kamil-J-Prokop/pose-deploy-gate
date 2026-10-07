from pathlib import Path

import pytest

from pose_deploy_gate.adapters import DummyAdapter
from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput, Keypoint, PosePrediction
from pose_deploy_gate.data import FileDataSource
from pose_deploy_gate.metrics import MetricsEngine
from pose_deploy_gate.runner import PredictionFailureKind, Runner
from pose_deploy_gate.runner.timing import Timer


def test_runner_results_produce_consistent_metrics(image_fixtures_dir: Path) -> None:
    fixture_count = len(tuple(image_fixtures_dir.rglob("*.jpg")))
    runner = Runner(
        adapter=DummyAdapter(),
        data_source=FileDataSource(
            input_dir=image_fixtures_dir,
            file_pattern="*.jpg",
            recursive=True,
        ),
    )

    run_result = runner.run()
    metrics = MetricsEngine().compute(run_result)

    assert metrics.errors.total_attempts == fixture_count
    assert metrics.errors.successful_predictions == fixture_count
    assert metrics.errors.failed_predictions == 0
    assert metrics.errors.error_rate == 0.0
    assert metrics.latency.sample_count == fixture_count

    assert metrics.latency.min_ns is not None
    assert metrics.latency.p50_ns is not None
    assert metrics.latency.p95_ns is not None
    assert metrics.latency.p99_ns is not None
    assert metrics.latency.max_ns is not None
    assert (
        0
        <= metrics.latency.min_ns
        <= metrics.latency.p50_ns
        <= metrics.latency.p95_ns
        <= metrics.latency.p99_ns
        <= metrics.latency.max_ns
    )


class ManualTimer(Timer):
    def __init__(self) -> None:
        self.current_ns = 0

    def now_ns(self) -> int:
        return self.current_ns


class ScriptedAdapter(PoseAdapter):
    def __init__(self, predictions: dict[str, tuple[AdapterOutput, int]], timer: ManualTimer):
        self.predictions = predictions
        self.timer = timer

    @property
    def name(self) -> str:
        return "scripted"

    def predict(self, image: ImageInput) -> AdapterOutput:
        output, duration_ns = self.predictions[image.image_id]
        self.timer.current_ns += duration_ns
        return output


SCHEMA = KeypointSchema(name="test_1", keypoint_names=("nose",))


def make_output(*, name="nose", x=0.5, confidence=0.9, person_id=None):
    return AdapterOutput(
        schema=SCHEMA,
        poses=(
            PosePrediction(
                keypoints=(Keypoint(name=name, x=x, y=0.5, confidence=confidence),),
                person_id=person_id,
            ),
        ),
    )


def run_predictions(tmp_path, predictions):
    for image_id in predictions:
        (tmp_path / f"{image_id}.jpg").touch()
    timer = ManualTimer()
    return Runner(
        adapter=ScriptedAdapter(predictions, timer),
        data_source=FileDataSource(input_dir=tmp_path, file_pattern="*.jpg"),
        timer=timer,
        warmup_iterations=0,
        continue_on_error=True,
    ).run()


def test_valid_prediction_contributes_latency_sample(tmp_path: Path) -> None:
    output = make_output()
    result = run_predictions(tmp_path, {"valid": (output, 100)})
    metrics = MetricsEngine().compute(result)

    assert result.predictions[0].output is output
    assert result.predictions[0].error is None
    assert metrics.errors.total_attempts == 1
    assert metrics.errors.successful_predictions == 1
    assert metrics.errors.failed_predictions == 0
    assert metrics.errors.error_rate == 0.0
    assert metrics.latency.sample_count == 1
    assert metrics.latency.min_ns == metrics.latency.mean_ns == metrics.latency.max_ns == 100
    assert metrics.latency.p50_ns == metrics.latency.p95_ns == metrics.latency.p99_ns == 100


@pytest.mark.parametrize(
    "invalid_output",
    [
        make_output(name="wrong"),
        make_output(x=-0.1),
        make_output(confidence=1.1),
        make_output(person_id=""),
    ],
    ids=["schema", "coordinates", "confidence", "identity"],
)
def test_invalid_output_counts_as_failure_without_latency_sample(tmp_path: Path, invalid_output):
    result = run_predictions(tmp_path, {"invalid": (invalid_output, 10_000)})
    metrics = MetricsEngine().compute(result)

    assert len(result.predictions) == 1
    prediction = result.predictions[0]
    assert prediction.error is not None
    assert prediction.failure_kind is PredictionFailureKind.OUTPUT_VALIDATION
    assert prediction.output is None
    assert prediction.timing.elapsed_ns == 10_000
    assert metrics.errors.total_attempts == 1
    assert metrics.errors.failed_predictions == 1
    assert metrics.errors.successful_predictions == 0
    assert metrics.errors.error_rate == 1.0
    assert metrics.latency.sample_count == 0
    assert metrics.latency.min_ns is None
    assert metrics.latency.mean_ns is None
    assert metrics.latency.max_ns is None
    assert metrics.latency.p50_ns is None
    assert metrics.latency.p95_ns is None
    assert metrics.latency.p99_ns is None


def test_mixed_validated_run_preserves_success_metrics_and_error_rate(tmp_path: Path) -> None:
    result = run_predictions(
        tmp_path,
        {
            "01-valid": (make_output(), 100),
            "02-invalid": (make_output(x=-0.1), 10_000),
            "03-valid": (make_output(), 300),
        },
    )
    metrics = MetricsEngine().compute(result)

    assert len(result.predictions) == 3
    assert result.predictions[1].failure_kind is PredictionFailureKind.OUTPUT_VALIDATION
    assert result.predictions[1].error is not None
    assert metrics.errors.total_attempts == 3
    assert metrics.errors.successful_predictions == 2
    assert metrics.errors.failed_predictions == 1
    assert metrics.errors.error_rate == pytest.approx(1 / 3)
    assert metrics.errors.success_rate == pytest.approx(2 / 3)
    assert metrics.latency.sample_count == 2
    assert metrics.latency.min_ns == 100
    assert metrics.latency.mean_ns == 200
    assert metrics.latency.max_ns == 300
    assert metrics.latency.p50_ns == 100
    assert metrics.latency.p95_ns == metrics.latency.p99_ns == 300
