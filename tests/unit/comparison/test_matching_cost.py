from math import sqrt

import pytest

from pose_deploy_gate.adapters.types import Keypoint, PosePrediction
from pose_deploy_gate.comparison import compute_pose_match_cost
from pose_deploy_gate.config.models import ComparisonConfig

NAMES = ("nose", "left_wrist", "right_wrist")


def _pose(coordinates, names=NAMES):
    return PosePrediction(
        keypoints=tuple(
            Keypoint(name=name, x=x, y=y) for name, (x, y) in zip(names, coordinates, strict=True)
        )
    )


def test_identical_poses_cost_zero():
    pose = _pose([(0, 0), (0.5, 0.5), (1, 1)])
    assert compute_pose_match_cost(pose, pose, NAMES, ComparisonConfig()) == 0


def test_translated_pose_cost_is_expected_and_symmetric():
    reference = _pose([(0, 0), (0.1, 0.2), (0.5, 0.5)])
    candidate = _pose([(0.3, 0.4), (0.4, 0.6), (0.8, 0.9)])
    config = ComparisonConfig()
    assert compute_pose_match_cost(reference, candidate, NAMES, config) == pytest.approx(
        0.5 / sqrt(2)
    )
    assert compute_pose_match_cost(candidate, reference, NAMES, config) == pytest.approx(
        0.5 / sqrt(2)
    )


def test_cost_is_mean_of_shared_distances():
    reference = _pose([(0, 0), (0, 0), (0, 0)])
    candidate = _pose([(0, 0), (0.5, 0.5), (1, 1)])
    assert compute_pose_match_cost(
        reference, candidate, NAMES, ComparisonConfig()
    ) == pytest.approx(0.5)


@pytest.mark.parametrize("missing", [(None, None), (None, 0.5), (0.5, None)])
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_missing_coordinates_excluded_from_cost(missing, side):
    incomplete = _pose([(0, 0), (0.5, 0.5), missing])
    complete = _pose([(0, 0), (1, 1), (0.2, 0.3)])
    reference, candidate = (incomplete, complete) if side == "reference" else (complete, incomplete)
    config = ComparisonConfig(min_pose_shared_keypoints=2, min_pose_shared_fraction=2 / 3)
    assert compute_pose_match_cost(reference, candidate, NAMES, config) == pytest.approx(0.25)


def test_insufficient_shared_count_is_unmatchable():
    pose = _pose([(0, 0), (0.5, 0.5), (None, None)])
    assert compute_pose_match_cost(pose, pose, NAMES, ComparisonConfig()) is None


def test_insufficient_shared_fraction_is_unmatchable():
    pose = _pose([(0, 0), (0.5, 0.5), (None, None)])
    config = ComparisonConfig(min_pose_shared_keypoints=2, min_pose_shared_fraction=0.7)
    assert compute_pose_match_cost(pose, pose, NAMES, config) is None


def test_only_compared_names_affect_cost_and_fraction_and_lookup_is_by_name():
    reference = _pose([(0, 0), (None, None), (1, 1)])
    candidate = _pose([(1, 1), (0, 0), (None, None)], names=("right_wrist", "nose", "left_wrist"))
    config = ComparisonConfig(min_pose_shared_keypoints=2, min_pose_shared_fraction=1)
    assert compute_pose_match_cost(reference, candidate, ("nose", "right_wrist"), config) == 0


def test_absent_named_point_counts_against_shared_fraction():
    pose = _pose([(0, 0)], names=("nose",))
    config = ComparisonConfig(min_pose_shared_keypoints=1, min_pose_shared_fraction=1)
    assert compute_pose_match_cost(pose, pose, ("nose", "left_wrist"), config) is None


@pytest.mark.parametrize("names", [(), NAMES])
def test_no_shared_points_is_unmatchable(names):
    pose = _pose([(None, None)] * 3)
    assert compute_pose_match_cost(pose, pose, names, ComparisonConfig()) is None


def test_matching_distance_threshold_is_left_to_assignment():
    reference = _pose([(0, 0)] * 3)
    candidate = _pose([(1, 1)] * 3)
    assert (
        compute_pose_match_cost(
            reference, candidate, NAMES, ComparisonConfig(max_pose_match_distance=0)
        )
        == 1
    )
