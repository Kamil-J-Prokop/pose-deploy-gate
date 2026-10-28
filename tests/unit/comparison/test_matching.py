from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from itertools import permutations, product
from math import fsum, nextafter
from pathlib import Path
from random import Random

import pytest

from pose_deploy_gate.adapters.dummy import DummyAdapter
from pose_deploy_gate.adapters.schema import DUMMY_5_SCHEMA, KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput, Keypoint, PosePrediction
from pose_deploy_gate.comparison import (
    PoseAssignment,
    PoseMatcher,
    PoseMatchingResult,
    compute_pose_match_cost,
    resolve_comparison_schema,
)
from pose_deploy_gate.config.models import ComparisonConfig


def _pose(position, *, schema=DUMMY_5_SCHEMA, missing=(), person_id=None):
    return PosePrediction(
        keypoints=tuple(
            Keypoint(
                name,
                None if name in missing else position,
                None if name in missing else position,
            )
            for name in schema.keypoint_names
        ),
        person_id=person_id,
    )


def _output(positions=(), *, schema=DUMMY_5_SCHEMA, ids=None):
    return AdapterOutput(
        schema,
        tuple(
            _pose(position, schema=schema, person_id=None if ids is None else ids[index])
            for index, position in enumerate(positions)
        ),
    )


def _match(reference, candidate, config=None):
    config = ComparisonConfig() if config is None else config
    schema = resolve_comparison_schema(
        reference.schema, candidate.schema, config.schema_mode, config.min_common_keypoints
    )
    return PoseMatcher().match(reference, candidate, schema, config)


def _pairs(result):
    return tuple(
        (match.reference_pose_index, match.candidate_pose_index) for match in result.pose_matches
    )


def _assert_partition(result, reference_count, candidate_count):
    reference_indices = [match.reference_pose_index for match in result.pose_matches]
    candidate_indices = [match.candidate_pose_index for match in result.pose_matches]
    assert reference_indices == sorted(set(reference_indices))
    assert len(candidate_indices) == len(set(candidate_indices))
    for matched, unmatched, count in (
        (reference_indices, result.unmatched_reference_pose_indices, reference_count),
        (candidate_indices, result.unmatched_candidate_pose_indices, candidate_count),
    ):
        assert isinstance(unmatched, tuple)
        assert list(unmatched) == sorted(set(unmatched))
        assert set(matched).isdisjoint(unmatched)
        assert sorted([*matched, *unmatched]) == list(range(count))
    for match in result.pose_matches:
        assert type(match.reference_pose_index) is int
        assert type(match.candidate_pose_index) is int
        assert type(match.matching_cost) is float


@pytest.mark.parametrize(
    "reference_positions,candidate_positions,expected",
    [
        ([], [], ()),
        ([], [0.1, 0.5, 0.9], ()),
        ([0.1, 0.5, 0.9], [], ()),
        ([0.2], [0.21], ((0, 0),)),
        ([0.5], [0.9, 0.51, 0.1], ((0, 1),)),
        ([0.9, 0.51, 0.1], [0.5], ((1, 0),)),
        ([0.1, 0.5, 0.9], [0.91, 0.11, 0.51], ((0, 1), (1, 2), (2, 0))),
        ([0.1, 0.3, 0.5, 0.7, 0.9], [0.71, 0.11, 0.91], ((0, 1), (3, 0), (4, 2))),
        ([0.71, 0.11, 0.91], [0.1, 0.3, 0.5, 0.7, 0.9], ((0, 3), (1, 0), (2, 4))),
    ],
)
def test_pose_counts_and_original_index_order(reference_positions, candidate_positions, expected):
    reference = _output(reference_positions)
    candidate = _output(candidate_positions)
    result = _match(reference, candidate)
    assert _pairs(result) == expected
    assert result.person_id_matching is False
    _assert_partition(result, len(reference_positions), len(candidate_positions))
    for match in result.pose_matches:
        assert match.matching_cost == pytest.approx(
            abs(
                reference_positions[match.reference_pose_index]
                - candidate_positions[match.candidate_pose_index]
            )
        )


def test_empty_side_does_not_call_solver(monkeypatch):
    def unexpected_solver_call(_):
        pytest.fail("Empty inputs must be handled without assignment")

    monkeypatch.setattr(
        "pose_deploy_gate.comparison.matching.linear_sum_assignment", unexpected_solver_call
    )
    for reference, candidate in (([], []), ([], [0.5]), ([0.5], [])):
        _assert_partition(
            _match(_output(reference), _output(candidate)), len(reference), len(candidate)
        )


def test_identical_poses_match_at_zero_threshold():
    result = _match(_output([0.5]), _output([0.5]), ComparisonConfig(max_pose_match_distance=0))
    assert result.pose_matches == (PoseAssignment(0, 0, 0.0),)


def test_all_pairs_beyond_threshold_are_unmatched():
    result = _match(_output([0.0, 0.1]), _output([0.8, 1.0]))
    assert result.pose_matches == ()
    assert result.person_id_matching is False
    _assert_partition(result, 2, 2)


def test_distance_at_maximum_is_accepted():
    result = _match(_output([0]), _output([1]), ComparisonConfig(max_pose_match_distance=1))
    assert result.pose_matches == (PoseAssignment(0, 0, 1.0),)


def test_geometric_greedy_failure_requires_global_assignment():
    # Row-first nearest matching takes 0.4 -> 0.5 (0.1), then 0.6 -> 0.1 (0.5).
    # The optimal complete assignment instead costs 0.3 + 0.1.
    result = _match(
        _output([0.4, 0.6]), _output([0.5, 0.1]), ComparisonConfig(max_pose_match_distance=1)
    )
    assert _pairs(result) == ((0, 1), (1, 0))
    assert fsum(match.matching_cost for match in result.pose_matches) == pytest.approx(0.4)


def test_crossing_people_keep_whole_pose_correspondence():
    # Both people have the same centroid; their wrists cross. Pairing individual
    # points or sorting people by centroid would lose the whole-body evidence.
    first = PosePrediction(
        tuple(
            Keypoint(name, x, y)
            for name, (x, y) in zip(
                DUMMY_5_SCHEMA.keypoint_names,
                [(0.3, 0.1), (0.8, 0.3), (0.6, 0.3), (0.3, 0.8), (0.5, 0.8)],
                strict=True,
            )
        )
    )
    second = replace(
        first, keypoints=tuple(replace(point, x=1 - point.x) for point in first.keypoints)
    )
    reference = AdapterOutput(DUMMY_5_SCHEMA, (first, second))
    candidate = AdapterOutput(
        DUMMY_5_SCHEMA,
        tuple(
            replace(
                pose, keypoints=tuple(replace(point, y=point.y + 0.01) for point in pose.keypoints)
            )
            for pose in (second, first)
        ),
    )
    assert _pairs(_match(reference, candidate)) == ((0, 1), (1, 0))


@pytest.mark.parametrize("missing_count", [1, 2])
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_default_shared_requirements_tolerate_missing_points(missing_count, side):
    complete = _output([0.5])
    partial = AdapterOutput(
        DUMMY_5_SCHEMA,
        (_pose(0.5, missing=DUMMY_5_SCHEMA.keypoint_names[:missing_count]),),
    )
    reference, candidate = (partial, complete) if side == "reference" else (complete, partial)
    assert _match(reference, candidate).pose_matches == (PoseAssignment(0, 0, 0.0),)


@pytest.mark.parametrize(
    "schema,missing,config",
    [
        (DUMMY_5_SCHEMA, DUMMY_5_SCHEMA.keypoint_names[:3], ComparisonConfig()),
        (
            KeypointSchema("six", (*DUMMY_5_SCHEMA.keypoint_names, "left_elbow")),
            DUMMY_5_SCHEMA.keypoint_names[:3],
            ComparisonConfig(min_pose_shared_fraction=0.6),
        ),
    ],
)
def test_shared_count_and_fraction_rejections(schema, missing, config):
    reference = _output([0.5], schema=schema)
    candidate = AdapterOutput(schema, (_pose(0.5, schema=schema, missing=missing),))
    result = _match(reference, candidate, config)
    assert result.pose_matches == ()
    _assert_partition(result, 1, 1)


def test_all_pairs_without_shared_points_are_unmatched():
    reference = _output([0.1, 0.5, 0.9])
    candidate = AdapterOutput(
        DUMMY_5_SCHEMA,
        tuple(_pose(0.5, missing=DUMMY_5_SCHEMA.keypoint_names) for _ in range(2)),
    )
    result = _match(reference, candidate)
    assert result.pose_matches == ()
    _assert_partition(result, 3, 2)


def test_a_pose_with_no_eligible_partner_does_not_block_other_matches():
    reference = _output([0.1, 0.9])
    candidate = AdapterOutput(
        DUMMY_5_SCHEMA,
        (_pose(0.9), _pose(0.1, missing=DUMMY_5_SCHEMA.keypoint_names), _pose(0.1)),
    )
    result = _match(reference, candidate)
    assert _pairs(result) == ((0, 2), (1, 0))
    assert result.unmatched_candidate_pose_indices == (1,)
    _assert_partition(result, 2, 3)


def test_different_schemas_use_only_common_names_in_cost_and_fraction():
    # Three common points suffice; neither model's additional points enter the
    # denominator. Candidate schema order differs from reference schema order.
    schema = KeypointSchema("candidate", ("right_wrist", "nose", "left_elbow", "left_wrist"))
    reference = _output([0.5])
    candidate = _output([0.5], schema=schema)
    candidate = replace(
        candidate,
        poses=(
            replace(
                candidate.poses[0],
                keypoints=tuple(
                    replace(point, x=1.0, y=1.0) if point.name == "left_elbow" else point
                    for point in candidate.poses[0].keypoints
                ),
            ),
        ),
    )
    result = _match(reference, candidate, ComparisonConfig(min_pose_shared_fraction=1))
    assert result.pose_matches == (PoseAssignment(0, 0, 0.0),)


def test_confidence_and_visibility_do_not_affect_matching():
    reference = _output([0.1, 0.9])
    candidate = _output([0.9, 0.1])
    candidate = replace(
        candidate,
        poses=tuple(
            replace(
                pose,
                confidence=0.0,
                keypoints=tuple(
                    replace(point, confidence=0.0, visible=False) for point in pose.keypoints
                ),
            )
            for pose in candidate.poses
        ),
    )
    result = _match(reference, candidate)
    assert _pairs(result) == ((0, 1), (1, 0))
    assert all(match.matching_cost == 0 for match in result.pose_matches)


@pytest.mark.parametrize(
    "reference_ids,candidate_ids,expected",
    [
        (("a", "b"), ("b", "a"), True),
        (("a", "b"), ("a", "b"), False),
        (("a", "b"), ("b", "different"), False),
        ((None, None), (None, None), False),
        (("a", "b"), ("b", None), False),
        ((None, "b"), ("b", "a"), False),
        (("a", "b"), ("b", "A"), False),
    ],
)
def test_person_id_matching_is_advisory_and_exact(reference_ids, candidate_ids, expected):
    result = _match(_output([0.1, 0.9], ids=reference_ids), _output([0.9, 0.1], ids=candidate_ids))
    assert _pairs(result) == ((0, 1), (1, 0))
    assert result.person_id_matching is expected


def test_unmatched_and_rejected_pairs_do_not_affect_id_diagnostic():
    result = _match(
        _output([0.1, 0.5, 0.9], ids=("a", None, "rejected-reference")),
        _output([0.11, 0.7], ids=("a", "rejected-candidate")),
    )
    assert _pairs(result) == ((0, 0),)
    assert result.person_id_matching is True
    _assert_partition(result, 3, 2)


def test_matching_ids_without_any_accepted_pair_return_false():
    result = _match(_output([0.0], ids=("a",)), _output([1.0], ids=("a",)))
    assert result.pose_matches == ()
    assert result.person_id_matching is False


def test_unique_correspondence_is_invariant_to_input_permutations():
    reference = _output([0.1, 0.5, 0.9])
    candidate = _output([0.51, 0.91, 0.11])
    expected = {(0, 2), (1, 0), (2, 1)}
    for reference_order, candidate_order in product(permutations(range(3)), repeat=2):
        result = _match(
            replace(reference, poses=tuple(reference.poses[index] for index in reference_order)),
            replace(candidate, poses=tuple(candidate.poses[index] for index in candidate_order)),
        )
        assert {(reference_order[i], candidate_order[j]) for i, j in _pairs(result)} == expected
        _assert_partition(result, 3, 3)


def test_matching_does_not_mutate_inputs_and_can_be_repeated():
    reference = _output([0.1, 0.9], ids=("a", "b"))
    candidate = _output([0.9, 0.1], ids=("b", "a"))
    schema = resolve_comparison_schema(reference.schema, candidate.schema, "common", 3)
    config = ComparisonConfig()
    before = deepcopy((reference, candidate, schema, config))
    matcher = PoseMatcher()
    result = matcher.match(reference, candidate, schema, config)
    matcher.match(_output(), candidate, schema, config)
    assert matcher.match(reference, candidate, schema, config) == result
    assert (reference, candidate, schema, config) == before


def test_matching_result_is_immutable_and_separate_from_keypoint_scoring():
    result = _match(_output([0.5]), _output([0.5]))
    assert isinstance(result, PoseMatchingResult)
    assert isinstance(result.pose_matches, tuple)
    with pytest.raises(FrozenInstanceError):
        result.person_id_matching = True
    with pytest.raises(FrozenInstanceError):
        result.pose_matches[0].matching_cost = 1
    assert not hasattr(result.pose_matches[0], "keypoints")


def test_dummy_adapter_outputs_can_be_matched_without_loading_images():
    image = ImageInput("dummy-image", Path("unused.jpg"))
    reference = DummyAdapter().predict(image)
    candidate = DummyAdapter(keypoint_confidence=0.2, pose_confidence=0.3).predict(image)
    result = _match(reference, candidate)
    assert result.pose_matches == (PoseAssignment(0, 0, 0.0),)
    assert result.person_id_matching is True


def _match_matrix(monkeypatch, costs, threshold=0.15):
    """Exercise the assignment independently of pose geometry and cost generation."""
    reference = _output([0.5] * len(costs))
    candidate = _output([0.5] * len(costs[0]))
    reference_indices = {id(pose): index for index, pose in enumerate(reference.poses)}
    candidate_indices = {id(pose): index for index, pose in enumerate(candidate.poses)}

    def cost(reference_pose, candidate_pose, _names, _config):
        return costs[reference_indices[id(reference_pose)]][candidate_indices[id(candidate_pose)]]

    monkeypatch.setattr("pose_deploy_gate.comparison.matching.compute_pose_match_cost", cost)
    return _match(reference, candidate, ComparisonConfig(max_pose_match_distance=threshold))


def test_maximum_valid_match_count_takes_priority_over_raw_cost(monkeypatch):
    # Raw minimization chooses 0.00 + 0.16, leaving only one accepted match.
    result = _match_matrix(monkeypatch, [[0.00, 0.14], [0.14, 0.16]])
    assert _pairs(result) == ((0, 1), (1, 0))
    assert [match.matching_cost for match in result.pose_matches] == [0.14, 0.14]


def test_smallest_edge_greedy_failure(monkeypatch):
    result = _match_matrix(monkeypatch, [[0.01, 0.02], [0.02, 0.14]])
    assert _pairs(result) == ((0, 1), (1, 0))


def test_minimum_valid_cost_breaks_equal_cardinality_tie(monkeypatch):
    result = _match_matrix(monkeypatch, [[0.12, 0.02], [0.03, 0.11], [None, None]])
    assert _pairs(result) == ((0, 1), (1, 0))
    assert result.unmatched_reference_pose_indices == (2,)
    assert fsum(match.matching_cost for match in result.pose_matches) == pytest.approx(0.05)


@pytest.mark.parametrize("cost", [0.0, 0.15, nextafter(0.15, 0.0), nextafter(0.15, 1.0), None, 1.0])
def test_threshold_is_inclusive_without_tolerance_or_penalty_leakage(monkeypatch, cost):
    result = _match_matrix(monkeypatch, [[cost]])
    if cost is not None and cost <= 0.15:
        assert result.pose_matches == (PoseAssignment(0, 0, cost),)
    else:
        assert result.pose_matches == ()
    _assert_partition(result, 1, 1)


def test_sentinel_must_outweigh_total_assignment_cost(monkeypatch):
    # Ten diagonal matches cost 10. A cyclic alternative uses nine zero-cost
    # pairs and one forbidden pair: a small fixed sentinel would prefer that.
    size = 10
    costs = [[None] * size for _ in range(size)]
    for index in range(size):
        costs[index][index] = 1.0
        if index < size - 1:
            costs[index][index + 1] = 0.0
    result = _match_matrix(monkeypatch, costs, threshold=1)
    assert _pairs(result) == tuple((index, index) for index in range(size))
    assert all(match.matching_cost == 1 for match in result.pose_matches)


@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 2)])
def test_ties_require_optimality_without_a_specific_identity_order(monkeypatch, shape):
    rows, columns = shape
    result = _match_matrix(monkeypatch, [[0.1] * columns for _ in range(rows)])
    assert len(result.pose_matches) == min(rows, columns)
    assert fsum(match.matching_cost for match in result.pose_matches) == pytest.approx(
        0.1 * min(rows, columns)
    )
    _assert_partition(result, rows, columns)


def _brute_force_objective(costs, threshold):
    """Enumerate all partial one-to-one matchings, without solver penalties."""
    best = (0, 0.0)
    for choices in product(range(-1, len(costs[0])), repeat=len(costs)):
        selected_columns = [column for column in choices if column >= 0]
        if len(selected_columns) != len(set(selected_columns)):
            continue
        distances = [costs[row][column] for row, column in enumerate(choices) if column >= 0]
        if any(cost is None or cost > threshold for cost in distances):
            continue
        best = min(best, (-len(distances), fsum(distances)))
    return best


def _assert_matches_brute_force(monkeypatch, costs, threshold):
    result = _match_matrix(monkeypatch, costs, threshold)
    expected_count, expected_cost = _brute_force_objective(costs, threshold)
    assert len(result.pose_matches) == -expected_count, costs
    assert fsum(match.matching_cost for match in result.pose_matches) == pytest.approx(
        expected_cost
    ), costs
    _assert_partition(result, len(costs), len(costs[0]))
    for match in result.pose_matches:
        original = costs[match.reference_pose_index][match.candidate_pose_index]
        assert original is not None and original <= threshold
        assert match.matching_cost == original


def test_every_two_by_two_cost_matrix_agrees_with_exhaustive_search(monkeypatch):
    for entries in product((None, 0.0, 0.07, 0.15, 0.16), repeat=4):
        costs = [entries[:2], entries[2:]]
        _assert_matches_brute_force(monkeypatch, costs, threshold=0.15)


@pytest.mark.parametrize("shape", [(1, 4), (4, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 4)])
@pytest.mark.parametrize("threshold", [0.0, 0.15, 1.0])
def test_small_rectangular_assignments_agree_with_exhaustive_search(monkeypatch, shape, threshold):
    random = Random(20261022)
    rows, columns = shape
    for _ in range(30):
        costs = [
            [random.choice((None, 0.0, 0.02, 0.14, 0.15, 0.16, 0.7, 1.0)) for _ in range(columns)]
            for _ in range(rows)
        ]
        _assert_matches_brute_force(monkeypatch, costs, threshold)


def test_returned_cost_is_original_shared_point_mean():
    reference = _output([0.5])
    candidate = AdapterOutput(
        DUMMY_5_SCHEMA,
        (
            replace(
                _pose(0.5, missing=("nose",)),
                keypoints=tuple(
                    replace(point, x=0.6, y=0.6) if point.name == "left_wrist" else point
                    for point in _pose(0.5, missing=("nose",)).keypoints
                ),
            ),
        ),
    )
    expected = compute_pose_match_cost(
        reference.poses[0], candidate.poses[0], DUMMY_5_SCHEMA.keypoint_names, ComparisonConfig()
    )
    assert _match(reference, candidate).pose_matches[0].matching_cost == expected
    assert expected == pytest.approx(0.025)
