import pytest

from pose_deploy_gate.adapters.schema import DUMMY_5_SCHEMA
from pose_deploy_gate.adapters.types import AdapterOutput, Keypoint, PosePrediction
from pose_deploy_gate.validation import AdapterOutputValidator, OutputValidationCode


def validate(keypoint_confidence=None, pose_confidence=None):
    output = AdapterOutput(
        schema=DUMMY_5_SCHEMA,
        poses=(
            PosePrediction(
                keypoints=(Keypoint(name="nose", x=0.5, y=0.5, confidence=keypoint_confidence),),
                confidence=pose_confidence,
            ),
        ),
    )
    return AdapterOutputValidator().validate_confidence(output)


def test_accepts_missing_keypoint_confidence():
    assert validate(pose_confidence=0.42) == []


def test_accepts_missing_pose_confidence():
    assert validate(keypoint_confidence=0.42) == []


@pytest.mark.parametrize("value", [0.0, 1.0, 0.42])
def test_accepts_confidence_boundaries(value):
    assert validate(value, value) == []


def test_rejects_negative_keypoint_confidence():
    issues = validate(keypoint_confidence=-0.0001)
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.CONFIDENCE_OUT_OF_RANGE
    assert issues[0].path == "poses[0].keypoints[0].confidence"


def test_rejects_keypoint_confidence_above_one():
    issues = validate(keypoint_confidence=1.0001)
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.CONFIDENCE_OUT_OF_RANGE
    assert issues[0].path == "poses[0].keypoints[0].confidence"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_keypoint_confidence(value):
    issues = validate(keypoint_confidence=value)
    assert len(issues) == 1
    assert issues[0].code == OutputValidationCode.NON_FINITE_CONFIDENCE
    assert issues[0].path == "poses[0].keypoints[0].confidence"


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (-0.0001, OutputValidationCode.CONFIDENCE_OUT_OF_RANGE),
        (1.0001, OutputValidationCode.CONFIDENCE_OUT_OF_RANGE),
        (float("nan"), OutputValidationCode.NON_FINITE_CONFIDENCE),
        (float("inf"), OutputValidationCode.NON_FINITE_CONFIDENCE),
        (float("-inf"), OutputValidationCode.NON_FINITE_CONFIDENCE),
    ],
)
def test_rejects_invalid_pose_confidence(value, code):
    issues = validate(pose_confidence=value)
    assert len(issues) == 1
    assert issues[0].code == code
    assert issues[0].path == "poses[0].confidence"


def test_accepts_zero_poses():
    output = AdapterOutput(schema=DUMMY_5_SCHEMA, poses=())
    assert AdapterOutputValidator().validate_confidence(output) == []


def test_collects_confidence_issues_across_all_poses_and_keypoints():
    valid = Keypoint(name="nose", x=0.5, y=0.5, confidence=0.42)
    invalid = Keypoint(name="left_wrist", x=0.5, y=0.5, confidence=float("nan"))
    output = AdapterOutput(
        schema=DUMMY_5_SCHEMA,
        poses=(
            PosePrediction(keypoints=(valid,), confidence=-0.1),
            PosePrediction(keypoints=(valid, invalid), confidence=1.1),
        ),
    )

    issues = AdapterOutputValidator().validate_confidence(output)

    assert [(issue.path, issue.code) for issue in issues] == [
        ("poses[0].confidence", OutputValidationCode.CONFIDENCE_OUT_OF_RANGE),
        ("poses[1].confidence", OutputValidationCode.CONFIDENCE_OUT_OF_RANGE),
        ("poses[1].keypoints[1].confidence", OutputValidationCode.NON_FINITE_CONFIDENCE),
    ]
