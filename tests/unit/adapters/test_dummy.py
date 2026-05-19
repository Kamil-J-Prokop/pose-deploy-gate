from pathlib import Path

from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.adapters.dummy import DummyAdapter
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput


def test_dummy_adapter_is_pose_adapter():
    assert isinstance(DummyAdapter(), PoseAdapter)


def test_dummy_adapter_name_is_dummy():
    assert DummyAdapter().name == "dummy"


def test_dummy_adapter_predict_returns_adapter_output():
    image = ImageInput(image_id="image-001", path=Path("/tmp/frame.jpg"))

    output = DummyAdapter().predict(image)

    assert isinstance(output, AdapterOutput)


def test_dummy_adapter_returns_deterministic_output():
    adapter = DummyAdapter()
    image = ImageInput(image_id="image-001", path=Path("/tmp/frame.jpg"))

    first = adapter.predict(image)
    second = adapter.predict(image)

    assert first == second


def test_dummy_adapter_output_contains_expected_keypoints():
    image = ImageInput(image_id="image-001", path=Path("/tmp/frame.jpg"))

    output = DummyAdapter().predict(image)

    assert len(output.poses) == 1
    assert tuple(keypoint.name for keypoint in output.poses[0].keypoints) == (
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )


def test_dummy_adapter_includes_image_id_in_metadata():
    image = ImageInput(image_id="image-123", path=Path("/tmp/frame.jpg"))

    output = DummyAdapter().predict(image)

    assert output.metadata["image_id"] == "image-123"


def test_dummy_adapter_accepts_confidence_params():
    image = ImageInput(image_id="image-001", path=Path("/tmp/frame.jpg"))
    adapter = DummyAdapter(keypoint_confidence=0.9, pose_confidence=0.8)

    output = adapter.predict(image)

    assert output.poses[0].confidence == 0.8
    assert all(keypoint.confidence == 0.9 for keypoint in output.poses[0].keypoints)
