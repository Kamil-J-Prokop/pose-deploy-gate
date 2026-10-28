from dataclasses import FrozenInstanceError

import pytest

from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.comparison import SchemaResolution, resolve_comparison_schema


def _schema(*names, name="test"):
    return KeypointSchema(name=name, keypoint_names=names)


def test_strict_accepts_equal_keypoints_with_different_schema_names():
    reference = _schema("nose", "left_wrist", name="reference")
    candidate = _schema("nose", "left_wrist", name="candidate")
    result = resolve_comparison_schema(reference, candidate, "strict", 1)
    assert result == SchemaResolution(reference, candidate, reference.keypoint_names, (), ())
    assert result.reference_schema is reference
    assert result.candidate_schema is candidate


@pytest.mark.parametrize("names", [("left_wrist", "nose"), ("nose",), ("nose", "right_wrist")])
def test_strict_rejects_mismatched_names_or_order(names):
    with pytest.raises(ValueError, match="identical keypoint names and order"):
        resolve_comparison_schema(_schema("nose", "left_wrist"), _schema(*names), "strict", 1)


def test_common_preserves_reference_order_and_reports_exclusions_in_schema_order():
    reference = _schema("nose", "shoulder_l", "shoulder_r", "wrist_l", "wrist_r")
    candidate = _schema("ankle_r", "shoulder_r", "nose", "ankle_l", "shoulder_l")
    result = resolve_comparison_schema(reference, candidate, "common", 3)
    assert result.reference_schema is reference
    assert result.candidate_schema is candidate
    assert result.compared_keypoints == ("nose", "shoulder_l", "shoulder_r")
    assert result.reference_only_keypoints == ("wrist_l", "wrist_r")
    assert result.candidate_only_keypoints == ("ankle_r", "ankle_l")


def test_common_accepts_identical_schemas():
    schema = _schema("nose")
    assert resolve_comparison_schema(schema, schema, "common", 1) == SchemaResolution(
        schema, schema, ("nose",), (), ()
    )


@pytest.mark.parametrize("candidate_names,minimum", [(("nose",), 2), (("ankle",), 1)])
def test_common_rejects_insufficient_intersection(candidate_names, minimum):
    with pytest.raises(ValueError, match="at least .* are required"):
        resolve_comparison_schema(
            _schema("nose", "wrist"), _schema(*candidate_names), "common", minimum
        )


def test_strict_requires_equality_without_common_mode_threshold():
    schema = _schema("nose")
    assert resolve_comparison_schema(schema, schema, "strict", 3).compared_keypoints == ("nose",)


def test_common_does_not_rename_aliases():
    reference = _schema("nose", "left_shoulder")
    candidate = _schema("nose", "shoulder_l")
    result = resolve_comparison_schema(reference, candidate, "common", 1)
    assert result.compared_keypoints == ("nose",)
    assert result.reference_only_keypoints == ("left_shoulder",)
    assert result.candidate_only_keypoints == ("shoulder_l",)


@pytest.mark.parametrize("mode", ["", "other", "STRICT"])
def test_rejects_unknown_mode(mode):
    schema = _schema("nose")
    with pytest.raises(ValueError, match="Unsupported schema mode"):
        resolve_comparison_schema(schema, schema, mode, 1)


@pytest.mark.parametrize("minimum", [0, -1, 1.5, True])
def test_rejects_invalid_minimum(minimum):
    schema = _schema("nose")
    with pytest.raises(ValueError, match="positive integer"):
        resolve_comparison_schema(schema, schema, "common", minimum)


def test_resolution_is_frozen():
    schema = _schema("nose")
    result = resolve_comparison_schema(schema, schema, "strict", 1)
    with pytest.raises(FrozenInstanceError):
        result.compared_keypoints = ()
