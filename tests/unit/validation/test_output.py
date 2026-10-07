import pytest

from pose_deploy_gate.adapters.schema import DUMMY_5_SCHEMA
from pose_deploy_gate.adapters.types import AdapterOutput, Keypoint, PosePrediction
from pose_deploy_gate.validation import AdapterOutputValidator, OutputValidationCode


def validate(x, y, visible=None):
    output = AdapterOutput(
        schema=DUMMY_5_SCHEMA,
        poses=(PosePrediction(keypoints=(Keypoint(name="nose", x=x, y=y, visible=visible),)),),
    )
    return AdapterOutputValidator().validate_keypoints(output)


def test_accepts_valid_normalized_coordinates():
    assert validate(0.25, 0.75) == []


@pytest.mark.parametrize(("x", "y"), [(0.0, 1.0), (1.0, 0.0)])
def test_accepts_boundary_coordinates(x, y):
    assert validate(x, y) == []


@pytest.mark.parametrize("visible", [None, False])
def test_accepts_missing_coordinate_pair(visible):
    assert validate(None, None, visible) == []


def test_rejects_only_x_missing():
    issues = validate(None, 0.5)
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.INVALID_COORDINATE_PAIR


def test_rejects_only_y_missing():
    issues = validate(0.5, None)
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.INVALID_COORDINATE_PAIR


@pytest.mark.parametrize("axis", ["x", "y"])
def test_rejects_negative_coordinate(axis):
    issues = validate(**{axis: -0.0001, "y" if axis == "x" else "x": 0.5})
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.COORDINATE_OUT_OF_RANGE
    assert issues[0].path == f"poses[0].keypoints[0].{axis}"


@pytest.mark.parametrize("axis", ["x", "y"])
def test_rejects_coordinate_above_one(axis):
    issues = validate(**{axis: 1.0001, "y" if axis == "x" else "x": 0.5})
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.COORDINATE_OUT_OF_RANGE
    assert issues[0].path == f"poses[0].keypoints[0].{axis}"


@pytest.mark.parametrize("axis", ["x", "y"])
def test_rejects_nan_coordinate(axis):
    issues = validate(**{axis: float("nan"), "y" if axis == "x" else "x": 0.5})
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.NON_FINITE_COORDINATE
    assert issues[0].path == f"poses[0].keypoints[0].{axis}"


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_rejects_infinite_coordinate(axis, value):
    issues = validate(**{axis: value, "y" if axis == "x" else "x": 0.5})
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.NON_FINITE_COORDINATE
    assert issues[0].path == f"poses[0].keypoints[0].{axis}"


@pytest.mark.parametrize(("x", "y"), [(None, None), (None, 0.5), (0.5, None)])
def test_rejects_visible_missing_keypoint(x, y):
    issues = validate(x, y, visible=True)
    assert issues[-1].code == OutputValidationCode.VISIBLE_MISSING_KEYPOINT
    assert issues[-1].path == "poses[0].keypoints[0].visible"
    assert len(issues) == (1 if x is None and y is None else 2)


def test_accepts_zero_poses():
    output = AdapterOutput(schema=DUMMY_5_SCHEMA, poses=())
    assert AdapterOutputValidator().validate_keypoints(output) == []


def test_collects_issues_across_all_poses_and_keypoints():
    valid = Keypoint(name="nose", x=0.5, y=0.5)
    invalid = Keypoint(name="left_wrist", x=-0.1, y=1.1)
    output = AdapterOutput(
        schema=DUMMY_5_SCHEMA,
        poses=(PosePrediction(keypoints=(valid,)), PosePrediction(keypoints=(valid, invalid))),
    )

    issues = AdapterOutputValidator().validate_keypoints(output)

    assert [issue.path for issue in issues] == [
        "poses[1].keypoints[1].x",
        "poses[1].keypoints[1].y",
    ]
    assert all(issue.code == OutputValidationCode.COORDINATE_OUT_OF_RANGE for issue in issues)
