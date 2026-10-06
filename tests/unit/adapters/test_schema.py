from dataclasses import FrozenInstanceError

import pytest

from pose_deploy_gate.adapters.schema import COCO_17_SCHEMA, DUMMY_5_SCHEMA, KeypointSchema


@pytest.mark.parametrize(
    ("field", "value"),
    [("name", "updated"), ("keypoint_names", ("left_eye",))],
)
def test_schema_is_immutable(field, value):
    schema = KeypointSchema(name="custom", keypoint_names=("nose",))

    with pytest.raises(FrozenInstanceError):
        setattr(schema, field, value)


@pytest.mark.parametrize("name", ["", " ", "\t\n"])
def test_schema_rejects_empty_name(name):
    with pytest.raises(ValueError):
        KeypointSchema(name=name, keypoint_names=("nose",))


def test_schema_rejects_empty_keypoint_list():
    with pytest.raises(ValueError):
        KeypointSchema(name="custom", keypoint_names=())


@pytest.mark.parametrize("empty_name", ["", " ", "\t\n"])
def test_schema_rejects_empty_keypoint_names(empty_name):
    with pytest.raises(ValueError):
        KeypointSchema(name="custom", keypoint_names=("nose", empty_name))


def test_schema_rejects_duplicate_keypoints():
    with pytest.raises(ValueError):
        KeypointSchema(name="custom", keypoint_names=("nose", "left_eye", "nose"))


def test_schema_accepts_custom_keypoints():
    schema = KeypointSchema(name="custom_3", keypoint_names=("head", "left_hand", "right_hand"))

    assert schema.name == "custom_3"
    assert schema.keypoint_names == ("head", "left_hand", "right_hand")


def test_dummy_schema_has_expected_keypoints():
    assert DUMMY_5_SCHEMA.name == "dummy_5"
    assert DUMMY_5_SCHEMA.keypoint_names == (
        "nose",
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
    )


def test_coco17_schema_has_expected_keypoints():
    assert COCO_17_SCHEMA.name == "coco_17"
    assert COCO_17_SCHEMA.keypoint_names == (
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
