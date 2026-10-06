"""Validation of normalized adapter output coordinates."""

from math import isfinite

from pose_deploy_gate.adapters.types import AdapterOutput
from pose_deploy_gate.validation.issues import OutputValidationCode, OutputValidationIssue


class AdapterOutputValidator:
    """Collect coordinate issues across every pose in an adapter output."""

    def validate_keypoints(self, output: AdapterOutput) -> list[OutputValidationIssue]:
        """Return all coordinate issues, or an empty list for valid output."""
        issues: list[OutputValidationIssue] = []
        for pose_index, pose in enumerate(output.poses):
            for keypoint_index, keypoint in enumerate(pose.keypoints):
                path = f"poses[{pose_index}].keypoints[{keypoint_index}]"
                if (keypoint.x is None) != (keypoint.y is None):
                    issues.append(
                        OutputValidationIssue(
                            path=path,
                            code=OutputValidationCode.INVALID_COORDINATE_PAIR,
                            message="x and y must both be present or both absent",
                        )
                    )
                for axis, value in (("x", keypoint.x), ("y", keypoint.y)):
                    if value is None:
                        continue
                    if not isfinite(value):
                        issues.append(
                            OutputValidationIssue(
                                path=f"{path}.{axis}",
                                code=OutputValidationCode.NON_FINITE_COORDINATE,
                                message=f"expected a finite coordinate, got {value}",
                            )
                        )
                    elif not 0 <= value <= 1:
                        issues.append(
                            OutputValidationIssue(
                                path=f"{path}.{axis}",
                                code=OutputValidationCode.COORDINATE_OUT_OF_RANGE,
                                message=f"expected [0, 1], got {value}",
                            )
                        )
                if keypoint.visible is True and (keypoint.x is None or keypoint.y is None):
                    issues.append(
                        OutputValidationIssue(
                            path=f"{path}.visible",
                            code=OutputValidationCode.VISIBLE_MISSING_KEYPOINT,
                            message="missing coordinates cannot have visible=True",
                        )
                    )
        return issues
