"""Per-image quality comparison of matched reference and candidate poses."""

from pose_deploy_gate.adapters.types import AdapterOutput
from pose_deploy_gate.comparison.distance import normalized_image_2d_distance
from pose_deploy_gate.comparison.matching import PoseMatcher
from pose_deploy_gate.comparison.result import ImageComparisonResult, KeypointComparison, PoseMatch
from pose_deploy_gate.comparison.schema import SchemaResolution
from pose_deploy_gate.config.models import ComparisonConfig


class QualityComparator:
    """Compare evaluation keypoints after matching whole people for one image."""

    def compare(
        self,
        image_id: str,
        reference: AdapterOutput,
        candidate: AdapterOutput,
        schema: SchemaResolution,
        config: ComparisonConfig,
    ) -> ImageComparisonResult:
        """Compare validated outputs for the same image and a compatible schema.

        Only points with both coordinates on both sides produce a deviation.
        Reference-present/candidate-missing flags record a missing event; a
        missing reference supplies no deviation sample or candidate penalty.
        Confidence and visibility do not affect presence. Inputs are not mutated.
        """
        matching = PoseMatcher().match(reference, candidate, schema, config)
        matches = []
        for assignment in matching.pose_matches:
            reference_points = {
                point.name: point
                for point in reference.poses[assignment.reference_pose_index].keypoints
            }
            candidate_points = {
                point.name: point
                for point in candidate.poses[assignment.candidate_pose_index].keypoints
            }
            keypoints = []
            for name in schema.compared_keypoints:
                reference_point = reference_points[name]
                candidate_point = candidate_points[name]
                reference_present = reference_point.x is not None and reference_point.y is not None
                candidate_present = candidate_point.x is not None and candidate_point.y is not None
                deviation = (
                    normalized_image_2d_distance(reference_point, candidate_point)
                    if reference_present and candidate_present
                    else None
                )
                keypoints.append(
                    KeypointComparison(name, reference_present, candidate_present, deviation)
                )
            matches.append(
                PoseMatch(
                    assignment.reference_pose_index,
                    assignment.candidate_pose_index,
                    assignment.matching_cost,
                    tuple(keypoints),
                )
            )
        return ImageComparisonResult(
            image_id,
            tuple(matches),
            matching.unmatched_reference_pose_indices,
            matching.unmatched_candidate_pose_indices,
        )
