"""Matching for reference comparison."""

from math import fsum

from pose_deploy_gate.adapters.types import PosePrediction
from pose_deploy_gate.comparison.distance import normalized_image_2d_distance
from pose_deploy_gate.config.models import ComparisonConfig


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
