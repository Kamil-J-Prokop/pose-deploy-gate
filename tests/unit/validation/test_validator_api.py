import pytest

import pose_deploy_gate.validation as validation
from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput, Keypoint, PosePrediction
from pose_deploy_gate.validation import (
    AdapterOutputValidationError,
    AdapterOutputValidator,
    OutputValidationCode,
    OutputValidationIssue,
)

SCHEMA = KeypointSchema(name="custom_1", keypoint_names=("nose",))


@pytest.fixture
def invalid_output():
    return AdapterOutput(
        schema=SCHEMA,
        poses=(
            PosePrediction(
                keypoints=(Keypoint(name="wrong", x=-0.1, y=1.1, confidence=-0.1),),
                person_id="",
                confidence=1.1,
            ),
        ),
    )


def valid_output(pose_count):
    return AdapterOutput(
        schema=SCHEMA,
        poses=tuple(
            PosePrediction(keypoints=(Keypoint(name="nose", x=0.5, y=0.5),))
            for _ in range(pose_count)
        ),
    )


@pytest.mark.parametrize("pose_count", [0, 1, 2])
def test_validate_returns_empty_tuple_for_valid_output(pose_count):
    assert AdapterOutputValidator().validate(valid_output(pose_count)) == ()


def test_validate_returns_all_detected_issues(invalid_output):
    issues = AdapterOutputValidator().validate(invalid_output)
    assert isinstance(issues, tuple)
    assert all(isinstance(issue, OutputValidationIssue) for issue in issues)
    assert [(issue.path, issue.code) for issue in issues] == [
        ("poses[0].keypoints", OutputValidationCode.KEYPOINT_SCHEMA_MISMATCH),
        ("poses[0].person_id", OutputValidationCode.EMPTY_PERSON_ID),
        ("poses[0].keypoints[0].x", OutputValidationCode.COORDINATE_OUT_OF_RANGE),
        ("poses[0].keypoints[0].y", OutputValidationCode.COORDINATE_OUT_OF_RANGE),
        ("poses[0].confidence", OutputValidationCode.CONFIDENCE_OUT_OF_RANGE),
        ("poses[0].keypoints[0].confidence", OutputValidationCode.CONFIDENCE_OUT_OF_RANGE),
    ]


@pytest.mark.parametrize("pose_count", [0, 1, 2])
def test_validate_or_raise_accepts_valid_output(pose_count):
    assert AdapterOutputValidator().validate_or_raise(valid_output(pose_count)) is None


def test_validate_or_raise_raises_for_invalid_output(invalid_output):
    validator = AdapterOutputValidator()
    expected_issues = validator.validate(invalid_output)
    assert len(expected_issues) == 6
    with pytest.raises(AdapterOutputValidationError) as exc:
        validator.validate_or_raise(invalid_output)
    assert exc.value.issues == expected_issues


def test_public_api_exports_expected_symbols():
    expected_exports = {
        "AdapterOutputValidationError": AdapterOutputValidationError,
        "AdapterOutputValidator": AdapterOutputValidator,
        "OutputValidationCode": OutputValidationCode,
        "OutputValidationIssue": OutputValidationIssue,
    }
    assert set(validation.__all__) == set(expected_exports)
    assert {name: getattr(validation, name) for name in validation.__all__} == expected_exports
