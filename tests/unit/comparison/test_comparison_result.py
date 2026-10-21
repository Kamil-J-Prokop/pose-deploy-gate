from dataclasses import FrozenInstanceError, fields

import pytest

from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.comparison import (
    CandidateComparisonResult,
    ImageComparisonResult,
    KeypointComparison,
    PoseMatch,
    resolve_comparison_schema,
)


@pytest.mark.parametrize(
    "reference_present,candidate_present,deviation",
    [
        (True, True, 0.0),
        (True, True, 0.25),
        (True, False, None),
        (False, True, None),
        (False, False, None),
    ],
)
def test_keypoint_comparison_stores_presence_and_optional_deviation(
    reference_present, candidate_present, deviation
):
    result = KeypointComparison("nose", reference_present, candidate_present, deviation)
    assert result.keypoint_name == "nose"
    assert result.reference_present is reference_present
    assert result.candidate_present is candidate_present
    assert result.deviation == deviation


def _candidate_result():
    reference = KeypointSchema("reference", ("nose", "left_wrist"))
    candidate = KeypointSchema("candidate", ("nose", "right_wrist"))
    resolution = resolve_comparison_schema(reference, candidate, "common", 1)
    keypoint = KeypointComparison("nose", True, True, 0.2)
    match = PoseMatch(2, 0, 0.2, (keypoint,))
    image = ImageComparisonResult("nested/image", (match,), (0, 1), (1, 2))
    return CandidateComparisonResult("production", "candidate-a", resolution, (image,))


def test_candidate_result_groups_schema_metadata_and_index_based_matches():
    result = _candidate_result()
    assert result.reference_model_id == "production"
    assert result.candidate_model_id == "candidate-a"
    assert result.schema_resolution.compared_keypoints == ("nose",)
    assert result.schema_resolution.reference_only_keypoints == ("left_wrist",)
    assert result.schema_resolution.candidate_only_keypoints == ("right_wrist",)
    image = result.images[0]
    assert image.image_id == "nested/image"
    assert image.unmatched_reference_pose_indices == (0, 1)
    assert image.unmatched_candidate_pose_indices == (1, 2)
    match = image.pose_matches[0]
    assert match.reference_pose_index == 2
    assert match.candidate_pose_index == 0
    assert match.matching_cost == 0.2
    assert match.keypoints == (KeypointComparison("nose", True, True, 0.2),)
    assert "person_id" not in {field.name for field in fields(match)}


@pytest.mark.parametrize(
    "reference_indices,candidate_indices", [((), ()), ((0, 1), ()), ((), (0,))]
)
def test_image_result_supports_no_matches(reference_indices, candidate_indices):
    result = ImageComparisonResult("empty", (), reference_indices, candidate_indices)
    assert result.pose_matches == ()
    assert result.unmatched_reference_pose_indices == reference_indices
    assert result.unmatched_candidate_pose_indices == candidate_indices


def test_candidate_result_supports_no_images():
    result = _candidate_result()
    empty = CandidateComparisonResult(
        result.reference_model_id, result.candidate_model_id, result.schema_resolution, ()
    )
    assert empty.images == ()


def test_result_hierarchy_is_frozen_and_uses_tuple_collections():
    candidate = _candidate_result()
    image = candidate.images[0]
    match = image.pose_matches[0]
    keypoint = match.keypoints[0]
    for collection in (
        candidate.images,
        image.pose_matches,
        image.unmatched_reference_pose_indices,
        image.unmatched_candidate_pose_indices,
        match.keypoints,
    ):
        assert isinstance(collection, tuple)
    for result in (candidate, image, match, keypoint):
        for field in fields(result):
            with pytest.raises(FrozenInstanceError):
                setattr(result, field.name, getattr(result, field.name))
