"""Results for reference comparison."""

from dataclasses import dataclass

from pose_deploy_gate.comparison.schema import SchemaResolution


@dataclass(frozen=True)
class PoseAssignment:
    """Person correspondence before final keypoint scoring, using original indices."""

    reference_pose_index: int
    candidate_pose_index: int
    matching_cost: float


@dataclass(frozen=True)
class PoseMatchingResult:
    """Assignments and unmatched indices for one pair of image outputs.

    Collections are ordered by original pose index. person_id_matching is an
    advisory observation: at least one match exists and every matched pair has
    supplied, identical IDs. Unmatched people do not participate in that flag.
    """

    pose_matches: tuple[PoseAssignment, ...]
    unmatched_reference_pose_indices: tuple[int, ...]
    unmatched_candidate_pose_indices: tuple[int, ...]
    person_id_matching: bool


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
    """Matched and unmatched pose indices for one reference-successful image.

    Evaluation records candidate failures with every reference pose unmatched
    and no candidate poses. The original failure remains in the model run.
    """

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
