"""Reference comparison package."""

from .aggregate import (
    KeypointQualityMetrics,
    QualityAggregator,
    QualityMetrics,
    aggregate_quality_metrics,
)
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
    "KeypointQualityMetrics",
    "PoseAssignment",
    "PoseMatch",
    "PoseMatcher",
    "PoseMatchingResult",
    "QualityAggregator",
    "QualityComparator",
    "QualityMetrics",
    "SchemaResolution",
    "aggregate_quality_metrics",
    "compute_pose_match_cost",
    "normalized_image_2d_distance",
    "resolve_comparison_schema",
]
