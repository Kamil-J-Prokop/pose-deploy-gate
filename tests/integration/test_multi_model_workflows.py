"""Config-driven regressions using synthetic adapters and the real evaluation pipeline."""

from dataclasses import dataclass, replace
from math import sqrt
from pathlib import Path

import pytest
import yaml

from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.adapters.exceptions import AdapterExecutionError
from pose_deploy_gate.adapters.factory import ADAPTER_REGISTRY
from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput, Keypoint, PosePrediction
from pose_deploy_gate.cli import build_parser, run
from pose_deploy_gate.config import load_config
from pose_deploy_gate.evaluation import EvaluationSession
from pose_deploy_gate.runner import PredictionFailureKind

SCHEMA = KeypointSchema(
    "synthetic_5", ("nose", "left_wrist", "right_wrist", "left_ankle", "right_ankle")
)
OFFSETS = {
    "nose": (0.0, -0.1),
    "left_wrist": (-0.04, 0.0),
    "right_wrist": (0.04, 0.0),
    "left_ankle": (-0.025, 0.15),
    "right_ankle": (0.025, 0.15),
    "reference_only": (0.0, 0.2),
    "candidate_only": (0.0, -0.2),
}


def pose(x: float, *, schema=SCHEMA, missing=()) -> PosePrediction:
    """Build distinct keypoints by name, without advisory person IDs."""
    return PosePrediction(
        tuple(
            Keypoint(name, None, None)
            if name in missing
            else Keypoint(name, x + OFFSETS[name][0], 0.4 + OFFSETS[name][1])
            for name in schema.keypoint_names
        )
    )


LEFT = pose(0.2)
RIGHT = pose(0.8)


def output(*poses: PosePrediction, schema=SCHEMA) -> AdapterOutput:
    return AdapterOutput(schema, tuple(poses))


class ScriptedPoseAdapter(PoseAdapter):
    """Return prescribed outputs or raise inference errors for each image ID."""

    def __init__(
        self,
        model_id: str,
        script: dict[str, AdapterOutput | AdapterExecutionError],
        calls: list[tuple[str, str]],
    ):
        self.model_id = model_id
        self.script = script
        self.calls = calls

    @property
    def name(self) -> str:
        return f"synthetic-{self.model_id}"

    def predict(self, image: ImageInput) -> AdapterOutput:
        self.calls.append((self.model_id, image.image_id))
        outcome = self.script[image.image_id]
        if isinstance(outcome, AdapterExecutionError):
            raise outcome
        return outcome


@dataclass(frozen=True)
class Workflow:
    session: EvaluationSession
    config_path: Path
    calls: list[tuple[str, str]]


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    """Replace only the adapter registry entry; keep config and evaluation real."""
    scripts = {}
    calls = []

    class RegisteredSyntheticAdapter(ScriptedPoseAdapter):
        def __init__(self, scenario: str):
            super().__init__(scenario, scripts[scenario], calls)

    monkeypatch.setitem(ADAPTER_REGISTRY, "dummy", RegisteredSyntheticAdapter)

    def build(models, *, schema_mode="strict", continue_on_error=False):
        scripts.update(models)
        input_dir = tmp_path / "images"
        # Every model uses one real file data source, including nested image IDs.
        for image_id in next(iter(models.values())):
            path = input_dir / f"{image_id}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        config = {
            "version": 1,
            "data": {"input_dir": str(input_dir), "file_pattern": "*.jpg", "recursive": True},
            "models": [
                {"id": model_id, "adapter": {"type": "dummy", "params": {"scenario": model_id}}}
                for model_id in models
            ],
            "runner": {"warmup_iterations": 0, "continue_on_error": continue_on_error},
            "comparison": {"schema_mode": schema_mode},
        }
        config_path = tmp_path / "evaluation.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return Workflow(EvaluationSession(load_config(config_path)), config_path, calls)

    return build


def test_identical_candidate_has_zero_quality_deviation(workflow):
    images = {"01-frame": output(LEFT), "nested/02-frame": output(RIGHT)}
    case = workflow({"production": images, "identical": dict(reversed(tuple(images.items())))})

    result = case.session.run()

    assert result.reference.model_id == "production"
    (candidate,) = result.candidates
    assert candidate.model.model_id == "identical"
    assert candidate.quality_metrics.matched_pose_count == 2
    assert candidate.quality_metrics.unmatched_reference_pose_count == 0
    assert candidate.quality_metrics.unmatched_candidate_pose_count == 0
    assert candidate.quality_metrics.keypoint_sample_count == 10
    assert candidate.quality_metrics.mean_keypoint_deviation == 0.0
    assert candidate.quality_metrics.p95_keypoint_deviation == 0.0
    assert candidate.quality_metrics.candidate_missing_keypoint_rate == 0.0
    assert candidate.quality_metrics.comparison_coverage == 1.0
    assert case.calls == [
        (model_id, image_id)
        for model_id in ("production", "identical")
        for image_id in ("01-frame", "nested/02-frame")
    ]


def test_three_candidates_run_once_in_config_order_and_share_reference_evidence(workflow):
    images = ("01-frame", "nested/02-frame")
    scripts = {
        "production": {name: output(LEFT, RIGHT) for name in images},
        "candidate-z": {name: output(LEFT, RIGHT) for name in images},
        "candidate-a": {name: output(RIGHT, LEFT) for name in images},
        "candidate-m": {name: output(RIGHT) for name in images},
    }
    case = workflow(scripts)

    result = case.session.run()

    assert [candidate.model.model_id for candidate in result.candidates] == [
        "candidate-z",
        "candidate-a",
        "candidate-m",
    ]
    assert case.calls == [(model_id, image_id) for model_id in scripts for image_id in images]
    for model in (result.reference, *(candidate.model for candidate in result.candidates)):
        assert model.metrics.errors.total_attempts == 2
        assert model.metrics.errors.failed_predictions == 0
        assert model.metrics.latency.sample_count == 2
    assert [candidate.quality_metrics.matched_pose_count for candidate in result.candidates] == [
        4,
        4,
        2,
    ]
    assert [
        candidate.quality_metrics.reference_pose_miss_rate for candidate in result.candidates
    ] == [0.0, 0.0, 0.5]
    assert all(
        candidate.quality_metrics.comparison_coverage == 1.0 for candidate in result.candidates
    )


def test_common_schema_matches_keypoints_by_name_and_records_exclusions(workflow):
    reference_schema = KeypointSchema(
        "reference_4", ("nose", "left_wrist", "right_wrist", "reference_only")
    )
    candidate_schema = KeypointSchema(
        "candidate_4", ("candidate_only", "right_wrist", "nose", "left_wrist")
    )
    case = workflow(
        {
            "production": {
                "frame": output(pose(0.3, schema=reference_schema), schema=reference_schema)
            },
            "candidate": {
                "frame": output(pose(0.3, schema=candidate_schema), schema=candidate_schema)
            },
        },
        schema_mode="common",
    )

    (candidate,) = case.session.run().candidates

    resolution = candidate.schema_resolution
    assert resolution.compared_keypoints == ("nose", "left_wrist", "right_wrist")
    assert resolution.reference_only_keypoints == ("reference_only",)
    assert resolution.candidate_only_keypoints == ("candidate_only",)
    assert candidate.quality_metrics.keypoint_sample_count == 3
    assert candidate.quality_metrics.mean_keypoint_deviation == 0.0
    assert [metric.name for metric in candidate.quality_metrics.per_keypoint] == list(
        resolution.compared_keypoints
    )
    assert [
        point.keypoint_name for point in candidate.image_comparisons[0].pose_matches[0].keypoints
    ] == list(resolution.compared_keypoints)


def test_strict_schema_rejection_stops_before_later_candidates(workflow):
    reordered_schema = KeypointSchema("reordered", tuple(reversed(SCHEMA.keypoint_names)))
    case = workflow(
        {
            "production": {"frame": output(LEFT)},
            "incompatible": {
                "frame": output(pose(0.2, schema=reordered_schema), schema=reordered_schema)
            },
            "later": {"frame": output(LEFT)},
        }
    )

    with pytest.raises(
        ValueError, match="Strict schema comparison requires identical keypoint names and order"
    ):
        case.session.run()

    assert case.calls == [("production", "frame"), ("incompatible", "frame")]


def test_shifted_single_person_has_known_normalized_deviation(workflow):
    case = workflow(
        {
            "production": {"frame": output(pose(0.3))},
            "shifted": {"frame": output(pose(0.38))},
        }
    )

    (candidate,) = case.session.run().candidates

    expected = 0.08 / sqrt(2)
    quality = candidate.quality_metrics
    assert quality.matched_pose_count == 1
    assert quality.reference_pose_miss_rate == 0.0
    assert quality.mean_keypoint_deviation == pytest.approx(expected)
    assert quality.p50_keypoint_deviation == pytest.approx(expected)
    assert quality.p95_keypoint_deviation == pytest.approx(expected)
    assert quality.max_keypoint_deviation == pytest.approx(expected)
    assert candidate.image_comparisons[0].pose_matches[0].matching_cost == pytest.approx(expected)


@pytest.mark.parametrize("reversed_order", [False, True], ids=["same-order", "reversed-order"])
def test_person_tuple_order_does_not_determine_matching(workflow, reversed_order):
    candidate_poses = (RIGHT, LEFT) if reversed_order else (LEFT, RIGHT)
    case = workflow(
        {
            "production": {"frame": output(LEFT, RIGHT)},
            "candidate": {"frame": output(*candidate_poses)},
        }
    )

    (candidate,) = case.session.run().candidates

    image = candidate.image_comparisons[0]
    expected = [(0, 1), (1, 0)] if reversed_order else [(0, 0), (1, 1)]
    assert [
        (match.reference_pose_index, match.candidate_pose_index) for match in image.pose_matches
    ] == expected
    assert all(person.person_id is None for person in candidate_poses)
    assert all(match.matching_cost == 0.0 for match in image.pose_matches)
    assert image.unmatched_reference_pose_indices == ()
    assert image.unmatched_candidate_pose_indices == ()
    assert candidate.quality_metrics.matched_pose_count == 2
    assert candidate.quality_metrics.keypoint_sample_count == 10
    assert candidate.quality_metrics.mean_keypoint_deviation == 0.0
    assert candidate.quality_metrics.p95_keypoint_deviation == 0.0
    assert candidate.quality_metrics.reference_pose_miss_rate == 0.0


def test_candidate_missing_person_is_a_quality_miss_not_an_inference_failure(workflow):
    case = workflow(
        {
            "production": {"frame": output(LEFT, RIGHT)},
            "candidate": {"frame": output(RIGHT)},
        }
    )

    (candidate,) = case.session.run().candidates

    image = candidate.image_comparisons[0]
    assert [
        (match.reference_pose_index, match.candidate_pose_index) for match in image.pose_matches
    ] == [(1, 0)]
    assert image.unmatched_reference_pose_indices == (0,)
    assert candidate.quality_metrics.unmatched_reference_pose_count == 1
    assert candidate.quality_metrics.reference_pose_miss_rate == 0.5
    assert candidate.quality_metrics.candidate_extra_pose_rate == 0.0
    assert candidate.quality_metrics.mean_keypoint_deviation == 0.0
    assert candidate.model.metrics.errors.error_rate == 0.0


def test_candidate_extra_person_keeps_original_candidate_index(workflow):
    case = workflow(
        {
            "production": {"frame": output(LEFT, RIGHT)},
            "candidate": {"frame": output(RIGHT, pose(0.5), LEFT)},
        }
    )

    (candidate,) = case.session.run().candidates

    image = candidate.image_comparisons[0]
    assert [
        (match.reference_pose_index, match.candidate_pose_index) for match in image.pose_matches
    ] == [(0, 2), (1, 0)]
    assert image.unmatched_candidate_pose_indices == (1,)
    assert candidate.quality_metrics.matched_pose_count == 2
    assert candidate.quality_metrics.unmatched_candidate_pose_count == 1
    assert candidate.quality_metrics.candidate_extra_pose_rate == pytest.approx(1 / 3)
    assert candidate.quality_metrics.reference_pose_miss_rate == 0.0
    assert candidate.quality_metrics.mean_keypoint_deviation == 0.0


def test_missing_keypoints_remain_missing_without_becoming_zero_deviation_samples(workflow):
    missing = ("left_ankle", "right_ankle")
    case = workflow(
        {
            "production": {"frame": output(pose(0.3))},
            "candidate": {"frame": output(pose(0.3, missing=missing))},
        }
    )

    (candidate,) = case.session.run().candidates

    quality = candidate.quality_metrics
    assert quality.matched_pose_count == 1
    assert quality.keypoint_sample_count == 3
    assert quality.candidate_missing_keypoint_rate == pytest.approx(2 / 5)
    assert quality.mean_keypoint_deviation == 0.0
    for metric in quality.per_keypoint:
        if metric.name in missing:
            assert metric.sample_count == 0
            assert metric.mean_deviation is None
            assert metric.missing_count == 1
            assert metric.missing_rate == 1.0
        else:
            assert metric.sample_count == 1
            assert metric.mean_deviation == 0.0
            assert metric.missing_rate == 0.0
    points = candidate.image_comparisons[0].pose_matches[0].keypoints
    assert all(point.reference_present for point in points)
    assert all(
        not point.candidate_present and point.deviation is None
        for point in points
        if point.keypoint_name in missing
    )


@pytest.mark.parametrize(
    "failed_model", ["production", "candidate"], ids=["reference-failure", "candidate-failure"]
)
@pytest.mark.parametrize("failure_kind", list(PredictionFailureKind))
def test_prediction_failures_preserve_runtime_evidence_and_quality_denominators(
    workflow, failed_model, failure_kind
):
    frames = ("01-success", "02-failure", "nested/03-success")
    scripts = {
        model: {name: output(LEFT, RIGHT) for name in frames}
        for model in ("production", "candidate")
    }
    if failure_kind is PredictionFailureKind.ADAPTER_EXECUTION:
        failure = AdapterExecutionError("synthetic inference failure")
    else:
        invalid_pose = replace(
            LEFT, keypoints=(replace(LEFT.keypoints[0], x=1.2), *LEFT.keypoints[1:])
        )
        failure = output(invalid_pose, RIGHT)
    scripts[failed_model]["02-failure"] = failure
    if failed_model == "production":
        # Candidate poses on reference-failed images must not be scored as extras.
        scripts["candidate"]["02-failure"] = output(LEFT, RIGHT, pose(0.5))
    case = workflow(scripts, continue_on_error=True)

    result = case.session.run()

    (candidate,) = result.candidates
    failed = result.reference if failed_model == "production" else candidate.model
    prediction = failed.run_result.predictions[1]
    assert prediction.output is None
    assert prediction.failure_kind is failure_kind
    assert (
        "synthetic inference failure"
        if failure_kind is PredictionFailureKind.ADAPTER_EXECUTION
        else "coordinate_out_of_range"
    ) in prediction.error
    assert failed.metrics.errors.total_attempts == 3
    assert failed.metrics.errors.failed_predictions == 1
    assert failed.metrics.errors.error_rate == pytest.approx(1 / 3)
    assert failed.metrics.latency.sample_count == 2
    healthy = candidate.model if failed_model == "production" else result.reference
    assert healthy.metrics.errors.failed_predictions == 0
    assert healthy.metrics.latency.sample_count == 3
    quality = candidate.quality_metrics
    assert quality.matched_pose_count == 4
    assert quality.mean_keypoint_deviation == 0.0
    assert quality.keypoint_sample_count == 20
    assert quality.unmatched_candidate_pose_count == 0
    if failed_model == "production":
        assert [image.image_id for image in candidate.image_comparisons] == [frames[0], frames[2]]
        assert quality.compared_image_count == 2
        assert quality.comparison_coverage == pytest.approx(2 / 3)
        assert quality.unmatched_reference_pose_count == 0
        assert quality.reference_pose_miss_rate == 0.0
    else:
        assert [image.image_id for image in candidate.image_comparisons] == list(frames)
        assert candidate.image_comparisons[1].unmatched_reference_pose_indices == (0, 1)
        assert quality.compared_image_count == 3
        assert quality.comparison_coverage == 1.0
        assert quality.unmatched_reference_pose_count == 2
        assert quality.reference_pose_miss_rate == pytest.approx(1 / 3)
    assert case.calls == [(model, image_id) for model in scripts for image_id in frames]


def test_reversed_person_order_is_zero_deviation_through_cli(workflow, capsys):
    case = workflow(
        {
            "production": {"frame": output(LEFT, RIGHT)},
            "reversed": {"frame": output(RIGHT, LEFT)},
        }
    )
    args = build_parser().parse_args(["--config", str(case.config_path)])

    assert run(args) == 0

    summary = capsys.readouterr().out
    assert "Reference: production" in summary
    assert "Candidate: reversed" in summary
    assert "Matched poses: 2" in summary
    assert "Missed reference poses: 0 (0.00%)" in summary
    assert "Mean keypoint deviation: 0.00%" in summary
    assert "P95 keypoint deviation: 0.00%" in summary
    assert "Comparison coverage: 100.00%" in summary
    assert case.calls == [("production", "frame"), ("reversed", "frame")]
