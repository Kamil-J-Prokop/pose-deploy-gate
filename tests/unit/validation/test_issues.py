from dataclasses import FrozenInstanceError

import pytest

from pose_deploy_gate.validation import OutputValidationCode, OutputValidationIssue


def test_validation_issue_is_immutable():
    issue = OutputValidationIssue(
        path="poses[0].keypoints[2].x",
        code=OutputValidationCode.COORDINATE_OUT_OF_RANGE,
        message="expected [0, 1], got 1.27",
    )

    with pytest.raises(FrozenInstanceError):
        issue.message = "updated"


def test_validation_codes_are_stable_strings():
    assert OutputValidationCode.COORDINATE_OUT_OF_RANGE.value == "coordinate_out_of_range"
    assert OutputValidationCode.INVALID_COORDINATE_PAIR.value == "invalid_coordinate_pair"
    assert str(OutputValidationCode.COORDINATE_OUT_OF_RANGE) == "coordinate_out_of_range"
    assert isinstance(OutputValidationCode.INVALID_COORDINATE_PAIR, str)
