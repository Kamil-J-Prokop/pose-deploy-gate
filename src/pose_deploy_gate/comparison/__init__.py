"""Reference comparison package."""

from .distance import normalized_image_2d_distance
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
    "normalized_image_2d_distance",
    "resolve_comparison_schema",
]
