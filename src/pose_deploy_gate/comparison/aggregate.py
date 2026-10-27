"""Aggregate reference-relative quality metrics across compared images."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum

from pose_deploy_gate.comparison.result import CandidateComparisonResult, ImageComparisonResult
from pose_deploy_gate.metrics.statistics import compute_percentiles


@dataclass(frozen=True)
class KeypointQualityMetrics:
    """Quality metrics for one resolved keypoint name."""

    name: str
    sample_count: int
    mean_deviation: float | None
    p95_deviation: float | None
    missing_count: int
    missing_rate: float | None


@dataclass(frozen=True)
class QualityMetrics:
    """Reference-relative quality metrics for one candidate model."""

    compared_image_count: int
    comparison_coverage: float | None
    matched_pose_count: int
    unmatched_reference_pose_count: int
    unmatched_candidate_pose_count: int
    reference_pose_miss_rate: float | None
    candidate_extra_pose_rate: float | None
    keypoint_sample_count: int
    mean_keypoint_deviation: float | None
    p50_keypoint_deviation: float | None
    p95_keypoint_deviation: float | None
    max_keypoint_deviation: float | None
    candidate_missing_keypoint_rate: float | None
    per_keypoint: tuple[KeypointQualityMetrics, ...]


class QualityAggregator:
    """Aggregate per-image comparisons without treating missing data as zero."""

    def aggregate(
        self,
        comparison: CandidateComparisonResult,
        total_image_count: int | None = None,
    ) -> QualityMetrics:
        """Return aggregate metrics for one candidate comparison.

        ``total_image_count`` includes images that could not be compared. If it
        is omitted, the supplied comparisons are treated as the complete image
        set. The schema resolution determines per-keypoint output order.
        """
        return aggregate_quality_metrics(comparison, total_image_count)


def aggregate_quality_metrics(
    comparison: CandidateComparisonResult,
    total_image_count: int | None = None,
) -> QualityMetrics:
    """Aggregate one candidate's image comparison results."""
    images = comparison.images
    compared_image_count = len(images)
    total_image_count = _resolve_total_image_count(compared_image_count, total_image_count)

    matched_pose_count = sum(len(image.pose_matches) for image in images)
    unmatched_reference_pose_count = sum(
        len(image.unmatched_reference_pose_indices) for image in images
    )
    unmatched_candidate_pose_count = sum(
        len(image.unmatched_candidate_pose_indices) for image in images
    )

    deviations = [
        keypoint.deviation
        for image in images
        for match in image.pose_matches
        for keypoint in match.keypoints
        if keypoint.deviation is not None
    ]
    reference_present_count = sum(
        keypoint.reference_present
        for image in images
        for match in image.pose_matches
        for keypoint in match.keypoints
    )
    candidate_missing_count = sum(
        keypoint.reference_present and not keypoint.candidate_present
        for image in images
        for match in image.pose_matches
        for keypoint in match.keypoints
    )

    p50, p95, _ = compute_percentiles(deviations)
    return QualityMetrics(
        compared_image_count=compared_image_count,
        comparison_coverage=_rate(compared_image_count, total_image_count),
        matched_pose_count=matched_pose_count,
        unmatched_reference_pose_count=unmatched_reference_pose_count,
        unmatched_candidate_pose_count=unmatched_candidate_pose_count,
        reference_pose_miss_rate=_rate(
            unmatched_reference_pose_count,
            matched_pose_count + unmatched_reference_pose_count,
        ),
        candidate_extra_pose_rate=_rate(
            unmatched_candidate_pose_count,
            matched_pose_count + unmatched_candidate_pose_count,
        ),
        keypoint_sample_count=len(deviations),
        mean_keypoint_deviation=_mean(deviations),
        p50_keypoint_deviation=p50,
        p95_keypoint_deviation=p95,
        max_keypoint_deviation=max(deviations, default=None),
        candidate_missing_keypoint_rate=_rate(candidate_missing_count, reference_present_count),
        per_keypoint=tuple(
            _aggregate_keypoint(images, name)
            for name in comparison.schema_resolution.compared_keypoints
        ),
    )


def _aggregate_keypoint(
    images: Sequence[ImageComparisonResult], name: str
) -> KeypointQualityMetrics:
    comparisons = [
        keypoint
        for image in images
        for match in image.pose_matches
        for keypoint in match.keypoints
        if keypoint.keypoint_name == name
    ]
    deviations = [keypoint.deviation for keypoint in comparisons if keypoint.deviation is not None]
    missing_count = sum(
        keypoint.reference_present and not keypoint.candidate_present for keypoint in comparisons
    )
    reference_present_count = sum(keypoint.reference_present for keypoint in comparisons)
    _, p95, _ = compute_percentiles(deviations)
    return KeypointQualityMetrics(
        name=name,
        sample_count=len(deviations),
        mean_deviation=_mean(deviations),
        p95_deviation=p95,
        missing_count=missing_count,
        missing_rate=_rate(missing_count, reference_present_count),
    )


def _resolve_total_image_count(compared_count: int, total_count: int | None) -> int:
    if total_count is None:
        return compared_count
    if isinstance(total_count, bool) or not isinstance(total_count, int) or total_count < 0:
        raise ValueError("total_image_count must be a non-negative integer")
    if total_count < compared_count:
        raise ValueError("total_image_count cannot be less than compared_image_count")
    return total_count


def _mean(values: Sequence[float]) -> float | None:
    return fsum(values) / len(values) if values else None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


__all__ = [
    "KeypointQualityMetrics",
    "QualityAggregator",
    "QualityMetrics",
    "aggregate_quality_metrics",
]
