from dataclasses import FrozenInstanceError

import pytest

from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.comparison import (
    CandidateComparisonResult,
    ImageComparisonResult,
    KeypointComparison,
    PoseMatch,
    QualityAggregator,
    aggregate_quality_metrics,
    resolve_comparison_schema,
)

SCHEMA = KeypointSchema("test", ("nose", "wrist"))
RESOLUTION = resolve_comparison_schema(SCHEMA, SCHEMA, "strict", 1)


def _point(name, deviation=0.0, *, reference_present=True, candidate_present=True):
    return KeypointComparison(name, reference_present, candidate_present, deviation)


def _match(*keypoints):
    return PoseMatch(0, 0, 0.0, tuple(keypoints))


def _image(image_id, matches=(), unmatched_reference=(), unmatched_candidate=()):
    return ImageComparisonResult(
        image_id, tuple(matches), tuple(unmatched_reference), tuple(unmatched_candidate)
    )


def _comparison(*images):
    return CandidateComparisonResult("reference", "candidate", RESOLUTION, tuple(images))


def test_perfect_comparison():
    metrics = QualityAggregator().aggregate(
        _comparison(
            _image(
                "one",
                [_match(_point("nose"), _point("wrist"))],
            )
        )
    )

    assert metrics.compared_image_count == 1
    assert metrics.comparison_coverage == 1.0
    assert metrics.matched_pose_count == 1
    assert metrics.reference_pose_miss_rate == 0.0
    assert metrics.candidate_extra_pose_rate == 0.0
    assert metrics.keypoint_sample_count == 2
    assert metrics.mean_keypoint_deviation == 0.0
    assert metrics.p50_keypoint_deviation == 0.0
    assert metrics.p95_keypoint_deviation == 0.0
    assert metrics.max_keypoint_deviation == 0.0
    assert metrics.candidate_missing_keypoint_rate == 0.0
    assert [(item.name, item.sample_count, item.missing_rate) for item in metrics.per_keypoint] == [
        ("nose", 1, 0.0),
        ("wrist", 1, 0.0),
    ]


def test_partial_differences_use_empirical_percentiles_and_reference_denominators():
    metrics = aggregate_quality_metrics(
        _comparison(
            _image(
                "one",
                [
                    _match(
                        _point("nose", 0.1),
                        _point(
                            "wrist",
                            None,
                            reference_present=True,
                            candidate_present=False,
                        ),
                    ),
                    _match(_point("nose", 0.3), _point("wrist", 0.2)),
                ],
            )
        ),
        total_image_count=2,
    )

    assert metrics.comparison_coverage == 0.5
    assert metrics.keypoint_sample_count == 3
    assert metrics.mean_keypoint_deviation == pytest.approx(0.2)
    assert metrics.p50_keypoint_deviation == 0.2
    assert metrics.p95_keypoint_deviation == 0.3
    assert metrics.max_keypoint_deviation == 0.3
    assert metrics.candidate_missing_keypoint_rate == 0.25
    assert metrics.per_keypoint[0].mean_deviation == pytest.approx(0.2)
    assert metrics.per_keypoint[0].p95_deviation == 0.3
    assert metrics.per_keypoint[1].missing_count == 1
    assert metrics.per_keypoint[1].missing_rate == 0.5


def test_all_candidate_keypoints_missing():
    missing = dict(deviation=None, reference_present=True, candidate_present=False)
    metrics = aggregate_quality_metrics(
        _comparison(_image("one", [_match(_point("nose", **missing), _point("wrist", **missing))]))
    )

    assert metrics.keypoint_sample_count == 0
    assert metrics.mean_keypoint_deviation is None
    assert metrics.p50_keypoint_deviation is None
    assert metrics.p95_keypoint_deviation is None
    assert metrics.max_keypoint_deviation is None
    assert metrics.candidate_missing_keypoint_rate == 1.0
    assert all(item.sample_count == 0 for item in metrics.per_keypoint)
    assert all(item.missing_rate == 1.0 for item in metrics.per_keypoint)


def test_unmatched_poses_contribute_to_reference_relative_rates():
    metrics = aggregate_quality_metrics(
        _comparison(
            _image(
                "one",
                [_match(_point("nose"), _point("wrist"))],
                unmatched_reference=(1, 2),
                unmatched_candidate=(1,),
            )
        )
    )

    assert metrics.matched_pose_count == 1
    assert metrics.unmatched_reference_pose_count == 2
    assert metrics.unmatched_candidate_pose_count == 1
    assert metrics.reference_pose_miss_rate == pytest.approx(2 / 3)
    assert metrics.candidate_extra_pose_rate == 0.5


def test_no_comparable_samples_produces_undefined_sample_metrics():
    unavailable = dict(deviation=None, reference_present=False, candidate_present=True)
    metrics = aggregate_quality_metrics(
        _comparison(
            _image("one", [_match(_point("nose", **unavailable), _point("wrist", **unavailable))])
        )
    )

    assert metrics.keypoint_sample_count == 0
    assert metrics.mean_keypoint_deviation is None
    assert metrics.candidate_missing_keypoint_rate is None
    assert all(
        item.missing_count == 0 and item.missing_rate is None for item in metrics.per_keypoint
    )


def test_aggregation_over_multiple_images():
    metrics = aggregate_quality_metrics(
        _comparison(
            _image("one", [_match(_point("nose", 0.1), _point("wrist", 0.2))]),
            _image(
                "two",
                [_match(_point("nose", 0.3), _point("wrist", 0.4))],
                unmatched_reference=(1,),
                unmatched_candidate=(1, 2),
            ),
        ),
        total_image_count=4,
    )

    assert metrics.compared_image_count == 2
    assert metrics.comparison_coverage == 0.5
    assert metrics.matched_pose_count == 2
    assert metrics.unmatched_reference_pose_count == 1
    assert metrics.unmatched_candidate_pose_count == 2
    assert metrics.keypoint_sample_count == 4
    assert metrics.mean_keypoint_deviation == pytest.approx(0.25)
    assert [item.sample_count for item in metrics.per_keypoint] == [2, 2]


def test_empty_aggregation_and_invalid_total_image_counts():
    comparison = _comparison()
    metrics = aggregate_quality_metrics(comparison)

    assert metrics.compared_image_count == 0
    assert metrics.comparison_coverage is None
    assert metrics.reference_pose_miss_rate is None
    assert metrics.candidate_extra_pose_rate is None
    assert metrics.per_keypoint[0].sample_count == 0
    with pytest.raises(FrozenInstanceError):
        metrics.matched_pose_count = 1

    for invalid in (-1, 0.5, True):
        with pytest.raises(ValueError, match="non-negative integer"):
            aggregate_quality_metrics(comparison, invalid)
    with pytest.raises(ValueError, match="less than compared"):
        aggregate_quality_metrics(_comparison(_image("one")), 0)
