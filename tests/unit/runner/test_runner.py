from pathlib import Path

from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput
from pose_deploy_gate.data.datasource import FileDataSource
from pose_deploy_gate.runner.runner import Runner


class FakeAdapter(PoseAdapter):
    def __init__(self) -> None:
        self.images_seen: list[ImageInput] = []

    @property
    def name(self) -> str:
        return "fake-adapter"

    def predict(self, image: ImageInput) -> AdapterOutput:
        self.images_seen.append(image)
        return AdapterOutput(poses=(), metadata={"image_id": image.image_id})


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
        poses=(), metadata={"image_id": "image-001"}
    )
    assert result.predictions[0].error is None
    assert result.predictions[1].image == images[1]
    assert result.predictions[1].output == AdapterOutput(
        poses=(), metadata={"image_id": "image-002"}
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
