from pathlib import Path

import pytest

from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.adapters.exceptions import AdapterExecutionError
from pose_deploy_gate.adapters.schema import DUMMY_5_SCHEMA
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput, Keypoint, PosePrediction
from pose_deploy_gate.data.datasource import FileDataSource
from pose_deploy_gate.runner.exceptions import RunnerExecutionError
from pose_deploy_gate.runner.result import PredictionFailureKind
from pose_deploy_gate.runner.runner import Runner
from pose_deploy_gate.validation import AdapterOutputValidationError, AdapterOutputValidator


class FakeAdapter(PoseAdapter):
    def __init__(self, fail_on_image_ids: set[str] | None = None) -> None:
        self.images_seen: list[ImageInput] = []
        self.fail_on_image_ids = fail_on_image_ids or set()

    @property
    def name(self) -> str:
        return "fake-adapter"

    def predict(self, image: ImageInput) -> AdapterOutput:
        self.images_seen.append(image)
        if image.image_id in self.fail_on_image_ids:
            raise AdapterExecutionError(f"prediction failed for {image.image_id}")
        return AdapterOutput(schema=DUMMY_5_SCHEMA, poses=(), metadata={"image_id": image.image_id})


class FakeDataSource(FileDataSource):
    def __init__(self, images: tuple[ImageInput, ...]) -> None:
        super().__init__(input_dir=Path("/tmp"))
        self._images = images

    def iter_images(self):  # type: ignore[override]
        yield from self._images


class FakeTimer:
    def __init__(self) -> None:
        self.current_ns = 0

    def now_ns(self) -> int:
        self.current_ns += 100
        return self.current_ns

    def elapsed_ns(self, start_ns: int) -> int:
        return self.now_ns() - start_ns


def _image(image_id: str) -> ImageInput:
    return ImageInput(image_id=image_id, path=Path(f"/tmp/{image_id}.jpg"))


def test_runner_warmup_calls_adapter_before_measured_predictions() -> None:
    images = (_image("image-001"), _image("image-002"), _image("image-003"))
    adapter = FakeAdapter()
    runner = Runner(
        adapter=adapter,
        data_source=FakeDataSource(images),
        warmup_iterations=2,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    runner.run()

    assert adapter.images_seen == [
        images[0],
        images[0],
        images[0],
        images[1],
        images[2],
    ]


def test_runner_zero_warmup_skips_warmup() -> None:
    images = (_image("image-001"), _image("image-002"))
    adapter = FakeAdapter()
    runner = Runner(
        adapter=adapter,
        data_source=FakeDataSource(images),
        warmup_iterations=0,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert adapter.images_seen == list(images)
    assert result.warmup.iterations == 0
    assert result.warmup.total_time_ns == 0


def test_runner_warmup_uses_first_image() -> None:
    images = (_image("image-003"), _image("image-001"), _image("image-002"))
    adapter = FakeAdapter()
    runner = Runner(
        adapter=adapter,
        data_source=FakeDataSource(images),
        warmup_iterations=2,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    runner.run()

    assert adapter.images_seen[:2] == [images[0], images[0]]


def test_runner_records_warmup_iterations() -> None:
    image = _image("image-001")
    runner = Runner(
        adapter=FakeAdapter(),
        data_source=FakeDataSource((image,)),
        warmup_iterations=3,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )
    empty_runner = Runner(
        adapter=FakeAdapter(),
        data_source=FakeDataSource(()),
        warmup_iterations=3,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()
    empty_result = empty_runner.run()

    assert result.warmup.iterations == 3
    assert empty_result.warmup.iterations == 0


def test_runner_records_warmup_total_time() -> None:
    runner = Runner(
        adapter=FakeAdapter(),
        data_source=FakeDataSource((_image("image-001"),)),
        warmup_iterations=3,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert result.warmup.total_time_ns == 100


def test_runner_does_not_include_warmup_in_measured_inference_time() -> None:
    images = (_image("image-001"), _image("image-002"))
    runner = Runner(
        adapter=FakeAdapter(),
        data_source=FakeDataSource(images),
        warmup_iterations=3,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert result.warmup.total_time_ns == 100
    assert result.measured_inference_time_ns == 200


def test_runner_returns_prediction_results() -> None:
    images = (_image("image-001"), _image("image-002"))
    runner = Runner(
        adapter=FakeAdapter(),
        data_source=FakeDataSource(images),
        warmup_iterations=1,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert len(result.predictions) == 2
    assert result.predictions[0].image == images[0]
    assert result.predictions[0].output == AdapterOutput(
        schema=DUMMY_5_SCHEMA, poses=(), metadata={"image_id": "image-001"}
    )
    assert result.predictions[0].error is None
    assert result.predictions[1].image == images[1]
    assert result.predictions[1].output == AdapterOutput(
        schema=DUMMY_5_SCHEMA, poses=(), metadata={"image_id": "image-002"}
    )
    assert result.predictions[1].error is None


def test_runner_captures_per_image_timing() -> None:
    images = (_image("image-001"), _image("image-002"))
    runner = Runner(
        adapter=FakeAdapter(),
        data_source=FakeDataSource(images),
        warmup_iterations=1,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert result.predictions[0].timing.image_id == "image-001"
    assert result.predictions[0].timing.elapsed_ns == 100
    assert result.predictions[1].timing.image_id == "image-002"
    assert result.predictions[1].timing.elapsed_ns == 100


def test_runner_total_time_includes_warmup_and_predictions() -> None:
    images = (_image("image-001"), _image("image-002"))
    runner = Runner(
        adapter=FakeAdapter(),
        data_source=FakeDataSource(images),
        warmup_iterations=3,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert result.warmup.total_time_ns == 100
    assert result.total_time_ns == 600


def test_runner_preserves_image_order_from_data_source() -> None:
    images = (_image("image-003"), _image("image-001"), _image("image-002"))
    runner = Runner(
        adapter=FakeAdapter(),
        data_source=FakeDataSource(images),
        warmup_iterations=1,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert tuple(prediction.image.image_id for prediction in result.predictions) == (
        "image-003",
        "image-001",
        "image-002",
    )


def test_runner_raises_when_prediction_fails_and_continue_on_error_is_false() -> None:
    images = (_image("image-001"), _image("image-002"))
    adapter = FakeAdapter(fail_on_image_ids={"image-001"})
    runner = Runner(
        adapter=adapter,
        data_source=FakeDataSource(images),
        warmup_iterations=0,
        continue_on_error=False,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    with pytest.raises(RunnerExecutionError, match="image-001"):
        runner.run()

    assert adapter.images_seen == [images[0]]


def test_runner_continues_when_prediction_fails_and_continue_on_error_is_true() -> None:
    images = (_image("image-001"), _image("image-002"))
    adapter = FakeAdapter(fail_on_image_ids={"image-001"})
    runner = Runner(
        adapter=adapter,
        data_source=FakeDataSource(images),
        warmup_iterations=0,
        continue_on_error=True,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert adapter.images_seen == list(images)
    assert len(result.predictions) == 2
    assert result.predictions[1].output is not None
    assert result.predictions[1].error is None


def test_runner_failed_prediction_contains_error_message() -> None:
    image = _image("image-001")
    runner = Runner(
        adapter=FakeAdapter(fail_on_image_ids={image.image_id}),
        data_source=FakeDataSource((image,)),
        warmup_iterations=0,
        continue_on_error=True,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert result.predictions[0].error == "prediction failed for image-001"


def test_adapter_failure_is_classified() -> None:
    images = (_image("failure"), _image("success"))
    runner = Runner(
        adapter=FakeAdapter(fail_on_image_ids={"failure"}),
        data_source=FakeDataSource(images),
        warmup_iterations=0,
        continue_on_error=True,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )
    result = runner.run()
    failure, success = result.predictions
    assert failure.failure_kind is PredictionFailureKind.ADAPTER_EXECUTION
    assert failure.error == "prediction failed for failure"
    assert failure.output is None
    assert failure.timing.elapsed_ns == 100
    assert success.error is None
    assert success.failure_kind is None
    assert success.output is not None


def test_runner_failed_prediction_has_no_output() -> None:
    image = _image("image-001")
    runner = Runner(
        adapter=FakeAdapter(fail_on_image_ids={image.image_id}),
        data_source=FakeDataSource((image,)),
        warmup_iterations=0,
        continue_on_error=True,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert result.predictions[0].output is None


def test_runner_failed_prediction_still_records_timing() -> None:
    image = _image("image-001")
    runner = Runner(
        adapter=FakeAdapter(fail_on_image_ids={image.image_id}),
        data_source=FakeDataSource((image,)),
        warmup_iterations=0,
        continue_on_error=True,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    result = runner.run()

    assert result.predictions[0].timing.image_id == image.image_id
    assert result.predictions[0].timing.elapsed_ns == 100


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_runner_warmup_failure_always_raises(continue_on_error: bool) -> None:
    image = _image("image-001")
    runner = Runner(
        adapter=FakeAdapter(fail_on_image_ids={image.image_id}),
        data_source=FakeDataSource((image,)),
        warmup_iterations=1,
        continue_on_error=continue_on_error,
        timer=FakeTimer(),  # type: ignore[arg-type]
    )

    with pytest.raises(RunnerExecutionError, match="warmup"):
        runner.run()


class OutputAdapter(FakeAdapter):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = iter(outputs)

    def predict(self, image):
        self.images_seen.append(image)
        return next(self.outputs)


class RecordingValidator(AdapterOutputValidator):
    def __init__(self, timer=None):
        self.outputs_seen = []
        self.timer = timer

    def validate_or_raise(self, output):
        self.outputs_seen.append(output)
        if self.timer is not None:
            self.timer.current_ns += 10_000
        super().validate_or_raise(output)


def _valid_output():
    return AdapterOutput(schema=DUMMY_5_SCHEMA, poses=())


def _invalid_output():
    return AdapterOutput(
        schema=DUMMY_5_SCHEMA,
        poses=(PosePrediction(keypoints=(Keypoint(name="nose", x=-0.1, y=0.5),)),),
    )


def test_runner_validates_successful_adapter_output():
    output = _valid_output()
    validator = RecordingValidator()
    result = Runner(
        adapter=OutputAdapter((output,)),
        data_source=FakeDataSource((_image("one"),)),
        warmup_iterations=0,
        validator=validator,
    ).run()
    assert validator.outputs_seen == [output]
    assert result.predictions[0].output is output
    assert result.predictions[0].failure_kind is None


@pytest.mark.parametrize("invalid", [False, True])
def test_runner_validation_does_not_change_inference_timing(invalid):
    timer = FakeTimer()
    output = _invalid_output() if invalid else _valid_output()
    result = Runner(
        adapter=OutputAdapter((output,)),
        data_source=FakeDataSource((_image("one"),)),
        warmup_iterations=0,
        continue_on_error=True,
        timer=timer,
        validator=RecordingValidator(timer),
    ).run()
    assert timer.current_ns == 10_300
    assert result.predictions[0].timing.elapsed_ns == 100
    assert result.measured_inference_time_ns == 100
    assert result.total_time_ns == 200


def test_runner_stops_on_invalid_output_by_default():
    images = (_image("invalid"), _image("valid"))
    adapter = OutputAdapter((_invalid_output(), _valid_output()))
    runner = Runner(adapter=adapter, data_source=FakeDataSource(images), warmup_iterations=0)
    with pytest.raises(
        RunnerExecutionError, match="Output validation failed for image invalid"
    ) as exc:
        runner.run()
    assert isinstance(exc.value.__cause__, AdapterOutputValidationError)
    assert adapter.images_seen == [images[0]]


def test_runner_continues_on_invalid_output_when_configured():
    images = (_image("invalid"), _image("valid"), _image("invalid-last"))
    valid = _valid_output()
    adapter = OutputAdapter((_invalid_output(), valid, _invalid_output()))
    result = Runner(
        adapter=adapter,
        data_source=FakeDataSource(images),
        warmup_iterations=0,
        continue_on_error=True,
        timer=FakeTimer(),
    ).run()
    assert adapter.images_seen == list(images)
    assert tuple(prediction.image for prediction in result.predictions) == images
    assert result.predictions[1].output is valid
    assert result.predictions[1].error is None
    assert result.successful_predictions == 1
    assert result.failed_predictions == 2
    assert result.total_time_ns == 600


def test_invalid_output_is_classified_as_validation_failure():
    output = _invalid_output()
    result = Runner(
        adapter=OutputAdapter((output,)),
        data_source=FakeDataSource((_image("invalid"),)),
        warmup_iterations=0,
        continue_on_error=True,
    ).run()
    prediction = result.predictions[0]
    assert prediction.failure_kind is PredictionFailureKind.OUTPUT_VALIDATION
    assert prediction.error == str(
        AdapterOutputValidationError(AdapterOutputValidator().validate(output))
    )


def test_invalid_output_is_not_preserved_as_success():
    result = Runner(
        adapter=OutputAdapter((_invalid_output(),)),
        data_source=FakeDataSource((_image("invalid"),)),
        warmup_iterations=0,
        continue_on_error=True,
    ).run()
    assert len(result.predictions) == 1
    assert result.predictions[0].output is None
    assert result.successful_predictions == 0
    assert result.failed_predictions == 1


@pytest.mark.parametrize("continue_on_error", [False, True])
@pytest.mark.parametrize("invalid_iteration", [0, 1])
def test_invalid_warmup_output_aborts_run(continue_on_error, invalid_iteration):
    outputs = [_valid_output(), _valid_output(), _valid_output()]
    outputs[invalid_iteration] = _invalid_output()
    adapter = OutputAdapter(outputs)
    validator = RecordingValidator()
    image = _image("one")
    runner = Runner(
        adapter=adapter,
        data_source=FakeDataSource((image,)),
        warmup_iterations=2,
        continue_on_error=continue_on_error,
        validator=validator,
    )
    with pytest.raises(RunnerExecutionError, match="warmup output validation failed") as exc:
        runner.run()
    assert isinstance(exc.value.__cause__, AdapterOutputValidationError)
    assert adapter.images_seen == [image] * (invalid_iteration + 1)
    assert validator.outputs_seen == outputs[: invalid_iteration + 1]


def test_runner_validates_every_warmup_output():
    outputs = tuple(_valid_output() for _ in range(3))
    validator = RecordingValidator()
    result = Runner(
        adapter=OutputAdapter(outputs),
        data_source=FakeDataSource((_image("one"),)),
        warmup_iterations=2,
        validator=validator,
    ).run()
    assert len(validator.outputs_seen) == 3
    assert all(
        actual is expected for actual, expected in zip(validator.outputs_seen, outputs, strict=True)
    )
    assert result.warmup.iterations == 2
    assert len(result.predictions) == 1
