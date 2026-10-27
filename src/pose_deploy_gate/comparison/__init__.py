"""Reference comparison package."""

from .comparator import QualityComparator
from .distance import normalized_image_2d_distance
from .matching import PoseMatcher, compute_pose_match_cost
from .result import (
    CandidateComparisonResult,
    ImageComparisonResult,
    KeypointComparison,
    PoseAssignment,
    PoseMatch,
    PoseMatchingResult,
)
from .schema import SchemaResolution, resolve_comparison_schema

__all__ = [
    "CandidateComparisonResult",
    "ImageComparisonResult",
    "KeypointComparison",
    "PoseAssignment",
    "PoseMatch",
    "PoseMatcher",
    "PoseMatchingResult",
    "QualityComparator",
    "SchemaResolution",
    "compute_pose_match_cost",
    "normalized_image_2d_distance",
    "resolve_comparison_schema",
]
