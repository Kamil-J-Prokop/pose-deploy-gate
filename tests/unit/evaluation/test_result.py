from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from pose_deploy_gate.adapters.schema import DUMMY_5_SCHEMA
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput
from pose_deploy_gate.comparison import (
    CandidateComparisonResult,
    ImageComparisonResult,
    QualityAggregator,
    resolve_comparison_schema,
)
from pose_deploy_gate.evaluation import (
    CandidateEvaluationResult,
    EvaluationResult,
    ModelEvaluationResult,
)
from pose_deploy_gate.metrics import MetricsEngine
from pose_deploy_gate.runner import (
    PredictionFailureKind,
    PredictionResult,
    PredictionTiming,
    RunResult,
    WarmupResult,
)


def _model(model_id, role, *, failed=False):
    prediction = PredictionResult(
        image=ImageInput("image", Path("image.jpg")),
        output=None if failed else AdapterOutput(DUMMY_5_SCHEMA, ()),
        timing=PredictionTiming("image", 100),
        error="adapter failed" if failed else None,
        failure_kind=PredictionFailureKind.ADAPTER_EXECUTION if failed else None,
    )
    run = RunResult(WarmupResult(0, 0), (prediction,), 150)
    return ModelEvaluationResult(model_id, role, run, MetricsEngine().compute(run))


def _candidate(reference, model, images):
    schema = resolve_comparison_schema(DUMMY_5_SCHEMA, DUMMY_5_SCHEMA, "strict", 1)
    raw = CandidateComparisonResult(reference.model_id, model.model_id, schema, images)
    quality = QualityAggregator().aggregate(raw, total_image_count=1)
    return CandidateEvaluationResult(model, schema, images, quality)


def test_evaluation_combines_execution_and_comparison_results_without_losing_failures():
    reference = _model("production", "reference")
    successful = _model("candidate-z", "candidate")
    failed = _model("candidate-a", "candidate", failed=True)
    image = ImageComparisonResult("image", (), (), ())
    compared = _candidate(reference, successful, (image,))
    uncomparable = _candidate(reference, failed, ())

    evaluation = EvaluationResult(reference, (compared, uncomparable))

    assert evaluation.reference is reference
    assert tuple(candidate.model.model_id for candidate in evaluation.candidates) == (
        "candidate-z",
        "candidate-a",
    )
    assert compared.model is successful
    assert compared.image_comparisons == (image,)
    assert compared.quality_metrics.comparison_coverage == 1.0
    assert uncomparable.model.run_result.predictions[0].error == "adapter failed"
    assert uncomparable.model.metrics.errors.error_rate == 1.0
    assert uncomparable.model.metrics.latency.mean_ns is None
    assert uncomparable.image_comparisons == ()
    assert uncomparable.quality_metrics.comparison_coverage == 0.0
    assert uncomparable.quality_metrics.mean_keypoint_deviation is None


def test_reference_only_evaluation():
    reference = _model("production", "reference")
    evaluation = EvaluationResult(reference, ())

    assert evaluation.reference is reference
    assert evaluation.candidates == ()


def test_evaluation_hierarchy_is_frozen():
    reference = _model("production", "reference")
    candidate = _candidate(reference, _model("candidate", "candidate"), ())
    evaluation = EvaluationResult(reference, (candidate,))

    for result in (reference, candidate.model, candidate, evaluation):
        for field in fields(result):
            with pytest.raises(FrozenInstanceError):
                setattr(result, field.name, getattr(result, field.name))
