"""Results for reference comparison."""

from dataclasses import dataclass

from pose_deploy_gate.comparison.schema import SchemaResolution


@dataclass(frozen=True)
class KeypointComparison:
    """Presence and normalized deviation for one resolved keypoint name.

    Presence means both coordinates are available. Deviation is None when
    either point is missing; a measured zero is a valid deviation.
    """

    keypoint_name: str
    reference_present: bool
    candidate_present: bool
    deviation: float | None


@dataclass(frozen=True)
class PoseMatch:
    """A matched pair identified by indices in the original outputs' pose tuples."""

    reference_pose_index: int
    candidate_pose_index: int
    matching_cost: float
    keypoints: tuple[KeypointComparison, ...]


@dataclass(frozen=True)
class ImageComparisonResult:
    """Matched and unmatched pose indices for one image."""

    image_id: str
    pose_matches: tuple[PoseMatch, ...]
    unmatched_reference_pose_indices: tuple[int, ...]
    unmatched_candidate_pose_indices: tuple[int, ...]


@dataclass(frozen=True)
class CandidateComparisonResult:
    """One candidate's image comparisons and schema metadata against a reference."""

    reference_model_id: str
    candidate_model_id: str
    schema_resolution: SchemaResolution
    images: tuple[ImageComparisonResult, ...]
