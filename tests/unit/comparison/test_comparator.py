from copy import deepcopy
from math import sqrt

import pytest

from pose_deploy_gate.adapters.schema import DUMMY_5_SCHEMA
from pose_deploy_gate.adapters.types import AdapterOutput, Keypoint, PosePrediction
from pose_deploy_gate.comparison import QualityComparator, resolve_comparison_schema
from pose_deploy_gate.config.models import ComparisonConfig


def _pose(position=0.2, *, missing=()):
    return PosePrediction(
        tuple(
            Keypoint(name, None, None)
            if name in missing
            else Keypoint(name, position, position, confidence=0.0, visible=False)
            for name in DUMMY_5_SCHEMA.keypoint_names
        )
    )


def _compare(reference_poses, candidate_poses):
    reference = AdapterOutput(DUMMY_5_SCHEMA, tuple(reference_poses))
    candidate = AdapterOutput(DUMMY_5_SCHEMA, tuple(candidate_poses))
    before = deepcopy((reference, candidate))
    schema = resolve_comparison_schema(DUMMY_5_SCHEMA, DUMMY_5_SCHEMA, "strict", 1)
    result = QualityComparator().compare(
        "nested/image", reference, candidate, schema, ComparisonConfig()
    )
    assert (reference, candidate) == before
    assert result.image_id == "nested/image"
    return result


def test_all_present_identical_pose():
    result = _compare([_pose()], [_pose()])
    (match,) = result.pose_matches
    assert match.matching_cost == 0.0
    assert tuple(point.keypoint_name for point in match.keypoints) == DUMMY_5_SCHEMA.keypoint_names
    assert all(point.reference_present and point.candidate_present for point in match.keypoints)
    assert all(point.deviation == 0.0 for point in match.keypoints)


def test_known_deviations_are_computed_per_keypoint():
    reference = _pose()
    candidate = PosePrediction(
        tuple(
            Keypoint(point.name, point.x + index * 0.02, point.y)
            for index, point in enumerate(reference.keypoints)
        )
    )
    (match,) = _compare([reference], [candidate]).pose_matches
    assert [point.deviation for point in match.keypoints] == pytest.approx(
        [index * 0.02 / sqrt(2) for index in range(5)]
    )


@pytest.mark.parametrize(
    "reference_missing,candidate_missing", [(False, True), (True, False), (True, True)]
)
def test_missing_keypoints_have_presence_flags_without_deviation(
    reference_missing, candidate_missing
):
    name = DUMMY_5_SCHEMA.keypoint_names[0]
    reference = _pose(missing=(name,) if reference_missing else ())
    candidate = _pose(missing=(name,) if candidate_missing else ())
    (match,) = _compare([reference], [candidate]).pose_matches
    point = match.keypoints[0]
    assert point.reference_present is (not reference_missing)
    assert point.candidate_present is (not candidate_missing)
    assert point.deviation is None
    assert all(point.deviation == 0.0 for point in match.keypoints[1:])


def test_multiple_matched_poses_use_assigned_indices_and_preserve_unmatched():
    result = _compare([_pose(0.1), _pose(0.5), _pose(0.9)], [_pose(0.52), _pose(0.11)])
    assert [
        (match.reference_pose_index, match.candidate_pose_index) for match in result.pose_matches
    ] == [(0, 1), (1, 0)]
    for match, expected in zip(result.pose_matches, (0.01, 0.02), strict=True):
        assert match.matching_cost == pytest.approx(expected)
        assert all(point.deviation == pytest.approx(expected) for point in match.keypoints)
    assert result.unmatched_reference_pose_indices == (2,)
    assert result.unmatched_candidate_pose_indices == ()


@pytest.mark.parametrize(
    "reference,candidate,unmatched_reference,unmatched_candidate",
    [([], [], (), ()), ([_pose()], [], (0,), ()), ([], [_pose()], (), (0,))],
)
def test_no_matched_poses(reference, candidate, unmatched_reference, unmatched_candidate):
    result = _compare(reference, candidate)
    assert result.pose_matches == ()
    assert result.unmatched_reference_pose_indices == unmatched_reference
    assert result.unmatched_candidate_pose_indices == unmatched_candidate


def test_common_schema_compares_only_resolved_names_in_reference_order():
    from pose_deploy_gate.adapters.schema import KeypointSchema

    reference_schema = KeypointSchema("reference", ("a", "b", "c", "reference_only"))
    candidate_schema = KeypointSchema("candidate", ("candidate_only", "c", "b", "a"))
    reference = AdapterOutput(
        reference_schema,
        (
            PosePrediction(
                tuple(Keypoint(name, 0.2, 0.2) for name in reference_schema.keypoint_names)
            ),
        ),
    )
    candidate = AdapterOutput(
        candidate_schema,
        (
            PosePrediction(
                tuple(Keypoint(name, 0.21, 0.21) for name in candidate_schema.keypoint_names)
            ),
        ),
    )
    schema = resolve_comparison_schema(reference_schema, candidate_schema, "common", 3)
    result = QualityComparator().compare("common", reference, candidate, schema, ComparisonConfig())
    (match,) = result.pose_matches
    assert tuple(point.keypoint_name for point in match.keypoints) == ("a", "b", "c")
    assert all(point.deviation == pytest.approx(0.01) for point in match.keypoints)
