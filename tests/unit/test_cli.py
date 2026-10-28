from argparse import Namespace
from dataclasses import replace
from unittest.mock import Mock

import pytest

from pose_deploy_gate.adapters import AdapterError, DummyAdapter
from pose_deploy_gate.adapters.schema import DUMMY_5_SCHEMA, KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput, Keypoint, PosePrediction
from pose_deploy_gate.cli import build_parser, run
from pose_deploy_gate.comparison import (
    CandidateComparisonResult,
    ImageComparisonResult,
    KeypointComparison,
    PoseMatch,
    QualityAggregator,
    resolve_comparison_schema,
)
from pose_deploy_gate.config.models import AppConfig
from pose_deploy_gate.data import DataSourceError
from pose_deploy_gate.evaluation import (
    CandidateEvaluationResult,
    EvaluationConsistencyError,
    EvaluationResult,
    ModelEvaluationResult,
)
from pose_deploy_gate.metrics import MetricsEngine
from pose_deploy_gate.runner import (
    PredictionFailureKind,
    PredictionResult,
    PredictionTiming,
    RunnerExecutionError,
    RunResult,
    WarmupResult,
)


def _args(tmp_path, *, list_inputs=False):
    return Namespace(
        config=tmp_path / "config.yaml", input=None, strict=False, list_inputs=list_inputs
    )


def _config(tmp_path, ids=("production", "candidate-z", "candidate-a"), **runner):
    return AppConfig(
        version=1,
        data={"input_dir": tmp_path, "file_pattern": "*.jpg"},
        models=[{"id": model_id, "adapter": {"type": "dummy"}} for model_id in ids],
        runner={"warmup_iterations": 0, **runner},
    )


def _model(model_id, role, tmp_path, *, elapsed_ns=1_000_000, failure_kind=None):
    predictions = tuple(
        PredictionResult(
            image=ImageInput(image_id, tmp_path / f"{image_id}.jpg"),
            output=None if failure_kind and index else AdapterOutput(DUMMY_5_SCHEMA, ()),
            timing=PredictionTiming(image_id, elapsed_ns),
            error="individual failure details" if failure_kind and index else None,
            failure_kind=failure_kind if index else None,
        )
        for index, image_id in enumerate(("a", "nested/b"))
    )
    result = RunResult(WarmupResult(0, 0), predictions, 20_000_000)
    return ModelEvaluationResult(model_id, role, result, MetricsEngine().compute(result))


def _evaluation(tmp_path, *, empty_poses=False):
    reference = _model(
        "production", "reference", tmp_path, failure_kind=PredictionFailureKind.ADAPTER_EXECUTION
    )
    schema = resolve_comparison_schema(
        KeypointSchema("reference_4", ("nose", "left", "right", "reference_only")),
        KeypointSchema("candidate_5", ("nose", "left", "right", "extra_1", "extra_2")),
        "common",
        3,
    )
    matches = (
        ()
        if empty_poses
        else (
            PoseMatch(
                0,
                0,
                0.01,
                (
                    KeypointComparison("nose", True, True, 0.01),
                    KeypointComparison("left", True, True, 0.02),
                    KeypointComparison("right", True, False, None),
                ),
            ),
        )
    )
    image = ImageComparisonResult(
        "a", matches, () if empty_poses else (1,), () if empty_poses else (1, 2)
    )
    candidates = []
    for model_id, latency, kind in (
        ("candidate-z", 3_000_000, PredictionFailureKind.ADAPTER_EXECUTION),
        ("candidate-a", 11_000_000, PredictionFailureKind.OUTPUT_VALIDATION),
    ):
        model = _model(model_id, "candidate", tmp_path, elapsed_ns=latency, failure_kind=kind)
        comparison = CandidateComparisonResult(reference.model_id, model_id, schema, (image,))
        quality = QualityAggregator().aggregate(comparison, total_image_count=2)
        candidates.append(CandidateEvaluationResult(model, schema, (image,), quality))
    return EvaluationResult(reference, tuple(candidates))


def _mock_session(monkeypatch, config, *, result=None, error=None):
    monkeypatch.setattr("pose_deploy_gate.cli.load_config", lambda path: config)
    session = Mock()
    session.run = Mock(return_value=result, side_effect=error)
    factory = Mock(return_value=session)
    monkeypatch.setattr("pose_deploy_gate.cli.EvaluationSession", factory)
    return factory, session


def test_cli_prints_every_model_once_in_order_with_its_runtime_metrics(
    tmp_path, monkeypatch, capsys
):
    config = _config(tmp_path)
    factory, session = _mock_session(monkeypatch, config, result=_evaluation(tmp_path))
    assert run(_args(tmp_path)) == 0
    output = capsys.readouterr().out
    factory.assert_called_once_with(config)
    session.run.assert_called_once_with()
    assert "PoseDeployGate evaluation completed." in output
    assert output.count("\nReference: production\n") == 1
    assert output.count("\nCandidate: candidate-z\n") == 1
    assert output.count("\nCandidate: candidate-a\n") == 1
    assert (
        output.index("Reference: production")
        < output.index("Candidate: candidate-z")
        < output.index("Candidate: candidate-a")
    )
    reference, first, second = output.split("\nCandidate:")
    assert "  Images: 2" in reference
    assert "  P95 latency: 1.000 ms" in reference
    assert "  Failure rate: 50.00%" in reference
    assert "  P95 latency: 3.000 ms" in first
    assert "  P95 latency: 11.000 ms" in second
    for section in (first, second):
        assert "  Failure rate: 50.00%" in section
        assert "  Failed predictions: 1" in section
    assert "    Adapter execution failures: 1" in first
    assert "    Output validation failures: 1" in second
    assert "individual failure details" not in output


def test_cli_prints_common_schema_exclusions_and_aggregate_quality(tmp_path, monkeypatch, capsys):
    _mock_session(monkeypatch, _config(tmp_path), result=_evaluation(tmp_path))
    assert run(_args(tmp_path)) == 0
    output = capsys.readouterr().out
    for line in (
        "    Reference: reference_4",
        "    Candidate: candidate_5",
        "    Compared keypoints: 3/4 reference, 3/5 candidate",
        "    Reference-only keypoints excluded: 1",
        "    Candidate-only keypoints excluded: 2",
        "    Comparison coverage: 50.00%",
        "    Matched poses: 1",
        "    Missed reference poses: 1 (50.00%)",
        "    Extra candidate poses: 2",
        "    Mean keypoint deviation: 1.50%",
        "    P95 keypoint deviation: 2.00%",
        "    Missing keypoint rate: 33.33%",
    ):
        assert output.count(line) == 2
    assert "nose" not in output
    assert "extra_1" not in output
    assert "per_keypoint" not in output


def test_cli_prints_undefined_quality_as_na(tmp_path, monkeypatch, capsys):
    _mock_session(monkeypatch, _config(tmp_path), result=_evaluation(tmp_path, empty_poses=True))
    assert run(_args(tmp_path)) == 0
    output = capsys.readouterr().out
    for line in (
        "Missed reference poses: 0 (n/a)",
        "Mean keypoint deviation: n/a",
        "P95 keypoint deviation: n/a",
        "Missing keypoint rate: n/a",
    ):
        assert line in output
    assert "Comparison coverage: 50.00%" in output
    assert "Mean keypoint deviation: 0.00%" not in output


def test_cli_reference_only_uses_session_and_lists_inputs_once(tmp_path, monkeypatch, capsys):
    model = _model("production", "reference", tmp_path)
    config = _config(tmp_path, ("production",))
    factory, session = _mock_session(monkeypatch, config, result=EvaluationResult(model, ()))
    assert run(_args(tmp_path, list_inputs=True)) == 0
    factory.assert_called_once_with(config)
    session.run.assert_called_once_with()
    output = capsys.readouterr().out
    assert "Reference: production" in output
    assert "Candidate:" not in output
    assert "Schema comparison:" not in output
    assert "  Failure rate: 0.00%" in output
    assert output.endswith("Input files:\n  001: a.jpg\n  002: nested/b.jpg\n")


@pytest.mark.parametrize(
    "error",
    [
        AdapterError("adapter failed"),
        DataSourceError("source failed"),
        RunnerExecutionError("runner failed"),
        EvaluationConsistencyError("missing image IDs: ['b']"),
        ValueError("Strict schema comparison requires identical keypoint names and order"),
        ValueError("Model 'production' has no successful output to resolve its schema"),
    ],
)
def test_cli_evaluation_errors_return_two_without_success_or_partial_summary(
    tmp_path, monkeypatch, capsys, error
):
    _mock_session(monkeypatch, _config(tmp_path), error=error)
    assert run(_args(tmp_path)) == 2
    output = capsys.readouterr().out
    assert f"ERROR: {error}" in output
    assert "evaluation completed" not in output
    assert "\nReference:" not in output
    assert "\nCandidate:" not in output


def test_cli_all_failed_single_model_returns_schema_error(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.jpg").touch()
    monkeypatch.setattr(
        DummyAdapter,
        "predict",
        lambda self, image: AdapterOutput(
            DUMMY_5_SCHEMA,
            (
                PosePrediction(
                    tuple(Keypoint(name, 1.27, 0.5) for name in DUMMY_5_SCHEMA.keypoint_names)
                ),
            ),
        ),
    )
    config = _config(tmp_path, ("production",), continue_on_error=True)
    monkeypatch.setattr("pose_deploy_gate.cli.load_config", lambda path: config)
    assert run(_args(tmp_path)) == 2
    output = capsys.readouterr().out
    assert "ERROR: Model 'production' has no successful output to resolve its schema" in output
    assert "evaluation completed" not in output


def test_cli_reports_actionable_fail_fast_validation_error(tmp_path, monkeypatch, capsys):
    (tmp_path / "person_004.jpg").touch()
    invalid_output = AdapterOutput(
        DUMMY_5_SCHEMA,
        (
            PosePrediction(
                tuple(
                    Keypoint(name, 1.27 if index == 0 else 0.5, 0.5)
                    for index, name in enumerate(DUMMY_5_SCHEMA.keypoint_names)
                )
            ),
        ),
    )
    monkeypatch.setattr(DummyAdapter, "predict", lambda self, image: invalid_output)
    monkeypatch.setattr(
        "pose_deploy_gate.cli.load_config", lambda path: _config(tmp_path, ("production",))
    )
    assert run(_args(tmp_path)) == 2
    output = capsys.readouterr().out
    assert "ERROR: Output validation failed for image person_004:" in output
    assert "poses[0].keypoints[0].x: coordinate_out_of_range: expected [0, 1], got 1.27" in output
    assert "evaluation completed" not in output


@pytest.mark.parametrize("flag", ["-v", "--verbose", "-verbose"])
def test_verbose_flag_aliases(flag):
    assert build_parser().parse_args([flag]).verbose is True
    assert build_parser().parse_args([]).verbose is False


def test_verbose_prints_full_runtime_statistics_for_every_model(tmp_path, monkeypatch, capsys):
    result = _evaluation(tmp_path)
    reference = _model("production", "reference", tmp_path)
    runtime = replace(
        reference.run_result,
        warmup=WarmupResult(3, 500_000),
        predictions=(
            reference.run_result.predictions[0],
            replace(
                reference.run_result.predictions[1], timing=PredictionTiming("nested/b", 3_000_000)
            ),
        ),
    )
    reference = replace(reference, run_result=runtime, metrics=MetricsEngine().compute(runtime))
    result = replace(result, reference=reference)
    factory, session = _mock_session(monkeypatch, _config(tmp_path), result=result)
    args = build_parser().parse_args(
        ["--config", str(tmp_path / "config.yaml"), "-v", "--list-inputs"]
    )

    assert run(args) == 0

    factory.assert_called_once()
    session.run.assert_called_once_with()
    output = capsys.readouterr().out
    reference_section, first, second = output.split("\nCandidate:")
    assert "  Warmup iterations: 3" in reference_section
    assert "  Warmup time: 0.500 ms" in reference_section
    assert "    Samples: 2" in reference_section
    for label, value in (
        ("Min", "1.000"),
        ("Mean", "2.000"),
        ("P50", "1.000"),
        ("P95", "3.000"),
        ("P99", "3.000"),
        ("Max", "3.000"),
    ):
        assert f"    {label}: {value} ms" in reference_section
    for section, latency in ((first, "3.000"), (second, "11.000")):
        assert f"    P50: {latency} ms" in section
        assert f"    P99: {latency} ms" in section
        assert "    Samples: 1" in section
        assert "    Error rate: 50.00%" in section
        assert "  Schema comparison:" in section
        assert "  Quality:" in section
    for section in (reference_section, first, second):
        assert "  Adapter: dummy" in section
        assert "  Latency:" in section
        assert "  Reliability:" in section
        assert "  Total runner time: 20.000 ms" in section
    assert "      Adapter execution failures: 1" in first
    assert "      Output validation failures: 1" in second
    assert output.count("Input files:") == 1


def test_default_summary_omits_verbose_statistics(tmp_path, monkeypatch, capsys):
    _mock_session(monkeypatch, _config(tmp_path), result=_evaluation(tmp_path))
    assert run(_args(tmp_path)) == 0
    output = capsys.readouterr().out
    for detail in (
        "  Adapter:",
        "Warmup iterations:",
        "  Latency:",
        "  Reliability:",
        "Total runner time:",
    ):
        assert detail not in output
    assert "  P95 latency:" in output
    assert "  Quality:" in output
