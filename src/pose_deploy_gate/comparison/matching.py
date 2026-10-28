"""Matching for reference comparison."""

from math import fsum

import numpy as np
from scipy.optimize import linear_sum_assignment

from pose_deploy_gate.adapters.types import AdapterOutput, PosePrediction
from pose_deploy_gate.comparison.distance import normalized_image_2d_distance
from pose_deploy_gate.comparison.result import PoseAssignment, PoseMatchingResult
from pose_deploy_gate.comparison.schema import SchemaResolution
from pose_deploy_gate.config.models import ComparisonConfig


class PoseMatcher:
    """Match whole people independently for one image, without using person IDs."""

    def match(
        self,
        reference: AdapterOutput,
        candidate: AdapterOutput,
        schema: SchemaResolution,
        config: ComparisonConfig,
    ) -> PoseMatchingResult:
        """Maximize valid pair count, then minimize total original matching cost.

        Inputs must be validated outputs for the same image, with a compatible
        resolved schema. A valid pair meets the shared-point requirements and
        has cost <= max_pose_match_distance. Confidence and visibility are not
        used. Equal optima have no identity tie-breaking guarantee.

        A rectangular assignment selects k = min(reference_count, candidate_count)
        pairs. Valid costs lie in [0, 1], so their total is at most k. Replacing
        every forbidden cost with k + 1 ensures that one fewer forbidden pair
        outweighs any change in the valid total. With that count fixed, the
        solver minimizes the original valid distances. Dummy rows are unnecessary.

        Rejection happens after assignment, using original costs, never penalties.
        The threshold must also influence the solver matrix: filtering a raw-cost
        assignment alone could discard a pair when more valid matches were possible.
        """
        # Match this image independently; tuple indices are not identities across frames.
        reference_count = len(reference.poses)
        candidate_count = len(candidate.poses)
        if not reference_count or not candidate_count:
            return PoseMatchingResult(
                pose_matches=(),
                unmatched_reference_pose_indices=tuple(range(reference_count)),
                unmatched_candidate_pose_indices=tuple(range(candidate_count)),
                person_id_matching=False,
            )

        # Rows are reference people, columns are candidates; None means insufficient shared points.
        original_costs = [
            [
                compute_pose_match_cost(
                    reference_pose, candidate_pose, schema.compared_keypoints, config
                )
                for candidate_pose in candidate.poses
            ]
            for reference_pose in reference.poses
        ]
        # A penalty above the maximum valid total makes valid-match count take priority.
        forbidden_cost = min(reference_count, candidate_count) + 1.0
        # Penalize above-threshold pairs before solving so they cannot displace more valid matches.
        solver_costs = np.array(
            [
                [
                    cost
                    if cost is not None and cost <= config.max_pose_match_distance
                    else forbidden_cost
                    for cost in row
                ]
                for row in original_costs
            ],
            dtype=float,
        )
        reference_indices, candidate_indices = linear_sum_assignment(solver_costs)

        matches = []
        for reference_index, candidate_index in zip(
            reference_indices.tolist(), candidate_indices.tolist(), strict=True
        ):
            # Reject forbidden filler assignments using original costs, never solver penalties.
            cost = original_costs[reference_index][candidate_index]
            if cost is None or cost > config.max_pose_match_distance:
                continue
            matches.append(PoseAssignment(reference_index, candidate_index, cost))
        matches.sort(key=lambda match: match.reference_pose_index)

        matched_reference = {match.reference_pose_index for match in matches}
        matched_candidate = {match.candidate_pose_index for match in matches}
        # require at least one accepted pair and supplied, equal IDs in every pair.
        person_id_matching = bool(matches) and all(
            reference.poses[match.reference_pose_index].person_id is not None
            and reference.poses[match.reference_pose_index].person_id
            == candidate.poses[match.candidate_pose_index].person_id
            for match in matches
        )
        # Only accepted assignments count as matched; report all remaining original indices.
        return PoseMatchingResult(
            pose_matches=tuple(matches),
            unmatched_reference_pose_indices=tuple(
                index for index in range(reference_count) if index not in matched_reference
            ),
            unmatched_candidate_pose_indices=tuple(
                index for index in range(candidate_count) if index not in matched_candidate
            ),
            person_id_matching=person_id_matching,
        )


def compute_pose_match_cost(
    reference_pose: PosePrediction,
    candidate_pose: PosePrediction,
    compared_keypoints: tuple[str, ...],
    config: ComparisonConfig,
) -> float | None:
    """Return mean shared-point distance for correspondence, or None if ineligible.

    The shared fraction uses all resolved compared names as its denominator.
    A point is shared only when both poses supply both coordinates. Inputs are
    expected to follow the validated adapter contract and resolved schema.
    This cost is not a final quality score. The assignment layer applies
    max_pose_match_distance; this function only enforces shared-point requirements.
    """
    if not compared_keypoints:
        return None

    reference_points = {point.name: point for point in reference_pose.keypoints}
    candidate_points = {point.name: point for point in candidate_pose.keypoints}
    shared_pairs = []
    for name in compared_keypoints:
        reference = reference_points.get(name)
        candidate = candidate_points.get(name)
        if reference is None or candidate is None:
            continue
        if any(value is None for value in (reference.x, reference.y, candidate.x, candidate.y)):
            continue
        shared_pairs.append((reference, candidate))

    shared_count = len(shared_pairs)
    if shared_count < config.min_pose_shared_keypoints:
        return None
    if shared_count / len(compared_keypoints) < config.min_pose_shared_fraction:
        return None

    return (
        fsum(
            normalized_image_2d_distance(reference, candidate)
            for reference, candidate in shared_pairs
        )
        / shared_count
    )
