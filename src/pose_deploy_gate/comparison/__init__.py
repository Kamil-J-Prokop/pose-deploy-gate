"""Reference comparison package."""

from .distance import normalized_image_2d_distance
from .matching import compute_pose_match_cost
from .result import (
    CandidateComparisonResult,
    ImageComparisonResult,
    KeypointComparison,
    PoseMatch,
)
from .schema import SchemaResolution, resolve_comparison_schema

__all__ = [
    "CandidateComparisonResult",
    "ImageComparisonResult",
    "KeypointComparison",
    "PoseMatch",
    "SchemaResolution",
    "compute_pose_match_cost",
    "normalized_image_2d_distance",
    "resolve_comparison_schema",
]
