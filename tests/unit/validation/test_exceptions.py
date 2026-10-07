from pose_deploy_gate.adapters.exceptions import AdapterError
from pose_deploy_gate.validation import (
    AdapterOutputValidationError,
    OutputValidationCode,
    OutputValidationIssue,
)


def test_validation_error_preserves_issues():
    issue = OutputValidationIssue(
        path="poses[0].keypoints[2].x",
        code=OutputValidationCode.COORDINATE_OUT_OF_RANGE,
        message="expected [0, 1], got 1.27",
    )
    issues = [issue]

    error = AdapterOutputValidationError(issues)
    issues.clear()

    assert error.issues == (issue,)
    assert error.issues[0] is issue
    assert isinstance(error, AdapterError)


def test_validation_error_formats_multiple_issues():
    error = AdapterOutputValidationError(
        (
            OutputValidationIssue(
                path="poses[0].keypoints[2].x",
                code=OutputValidationCode.COORDINATE_OUT_OF_RANGE,
                message="expected [0, 1], got 1.27",
            ),
            OutputValidationIssue(
                path="poses[0].keypoints[4]",
                code=OutputValidationCode.INVALID_COORDINATE_PAIR,
                message="x and y must both be present or both absent",
            ),
        )
    )

    assert str(error) == (
        "Adapter output validation failed with 2 issues:\n"
        "- poses[0].keypoints[2].x: coordinate_out_of_range: expected [0, 1], got 1.27\n"
        "- poses[0].keypoints[4]: invalid_coordinate_pair: "
        "x and y must both be present or both absent"
    )
