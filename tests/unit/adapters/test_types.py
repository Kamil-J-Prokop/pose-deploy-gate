from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput, Keypoint, PosePrediction


def test_image_input_stores_id_and_path():
    image = ImageInput(image_id="image-001", path=Path("/tmp/frame.jpg"))

    assert image.image_id == "image-001"
    assert image.path == Path("/tmp/frame.jpg")


def test_keypoint_is_immutable():
    keypoint = Keypoint(name="nose", x=12.5, y=18.0, confidence=0.97)

    with pytest.raises(FrozenInstanceError):
        keypoint.x = 20.0


def test_pose_prediction_contains_keypoints():
    keypoints = (
        Keypoint(name="nose", x=10.0, y=20.0, confidence=0.95),
        Keypoint(name="left_eye", x=8.0, y=18.0, confidence=0.91),
    )

    pose = PosePrediction(keypoints=keypoints, confidence=0.93)

    assert pose.keypoints == keypoints
    assert pose.confidence == 0.93


def test_adapter_output_contains_poses_and_metadata():
    pose = PosePrediction(
        keypoints=(Keypoint(name="nose", x=10.0, y=20.0, confidence=0.95),),
        confidence=0.93,
    )

    output = AdapterOutput(
        poses=(pose,),
        metadata={"model_name": "dummy", "latency_ms": 12.4},
    )

    assert output.poses == (pose,)
    assert output.metadata == {"model_name": "dummy", "latency_ms": 12.4}
