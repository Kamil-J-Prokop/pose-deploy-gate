"""Validation of normalized adapter output values."""

from math import isfinite

from pose_deploy_gate.adapters.types import AdapterOutput
from pose_deploy_gate.validation.issues import OutputValidationCode, OutputValidationIssue


class AdapterOutputValidator:
    """Collect schema, identity, coordinate, and confidence issues across an adapter output."""

    def validate_person_ids(self, output: AdapterOutput) -> list[OutputValidationIssue]:
        """Return issues for empty or duplicate person IDs within one output."""
        issues: list[OutputValidationIssue] = []
        seen_ids: set[str] = set()
        for pose_index, pose in enumerate(output.poses):
            if pose.person_id is None:
                continue
            path = f"poses[{pose_index}].person_id"
            if pose.person_id == "":
                issues.append(
                    OutputValidationIssue(
                        path=path,
                        code=OutputValidationCode.EMPTY_PERSON_ID,
                        message="expected a non-empty person ID",
                    )
                )
            elif pose.person_id in seen_ids:
                issues.append(
                    OutputValidationIssue(
                        path=path,
                        code=OutputValidationCode.DUPLICATE_PERSON_ID,
                        message=f"duplicate person ID {pose.person_id!r}",
                    )
                )
            else:
                seen_ids.add(pose.person_id)
        return issues

    def validate_schema(self, output: AdapterOutput) -> list[OutputValidationIssue]:
        """Return issues for keypoint names that differ from the declared schema."""
        issues: list[OutputValidationIssue] = []
        expected_names = output.schema.keypoint_names

        for pose_index, pose in enumerate(output.poses):
            actual_names = tuple(keypoint.name for keypoint in pose.keypoints)
            if actual_names != expected_names:
                issues.append(
                    OutputValidationIssue(
                        path=f"poses[{pose_index}].keypoints",
                        code=OutputValidationCode.KEYPOINT_SCHEMA_MISMATCH,
                        message=(
                            f"expected keypoint names {expected_names!r}, got {actual_names!r}"
                        ),
                    )
                )
        return issues

    def validate_confidence(self, output: AdapterOutput) -> list[OutputValidationIssue]:
        """Return confidence issues for every pose and keypoint."""
        issues: list[OutputValidationIssue] = []
        for pose_index, pose in enumerate(output.poses):
            path = f"poses[{pose_index}]"
            issues.extend(self._validate_confidence_value(pose.confidence, f"{path}.confidence"))
            for keypoint_index, keypoint in enumerate(pose.keypoints):
                issues.extend(
                    self._validate_confidence_value(
                        keypoint.confidence,
                        f"{path}.keypoints[{keypoint_index}].confidence",
                    )
                )
        return issues

    @staticmethod
    def _validate_confidence_value(value: float | None, path: str) -> list[OutputValidationIssue]:
        if value is None:
            return []
        if not isfinite(value):
            return [
                OutputValidationIssue(
                    path=path,
                    code=OutputValidationCode.NON_FINITE_CONFIDENCE,
                    message=f"expected a finite confidence, got {value}",
                )
            ]
        if not 0 <= value <= 1:
            return [
                OutputValidationIssue(
                    path=path,
                    code=OutputValidationCode.CONFIDENCE_OUT_OF_RANGE,
                    message=f"expected [0, 1], got {value}",
                )
            ]
        return []

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
