from pathlib import Path
from unittest.mock import Mock

import pytest

from pose_deploy_gate.adapters.schema import DUMMY_5_SCHEMA, KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput, Keypoint, PosePrediction
from pose_deploy_gate.comparison import QualityComparator
from pose_deploy_gate.config.models import AppConfig
from pose_deploy_gate.data import FileDataSource
from pose_deploy_gate.evaluation import EvaluationConsistencyError, EvaluationSession
from pose_deploy_gate.evaluation import session as session_module
from pose_deploy_gate.metrics import MetricsEngine
from pose_deploy_gate.runner import (
    PredictionFailureKind,
    PredictionResult,
    PredictionTiming,
    Runner,
    RunnerExecutionError,
    RunResult,
    WarmupResult,
)


def _config(tmp_path, ids=("production", "candidate-z", "candidate-a"), **comparison):
    return AppConfig(
        version=1,
        data={"input_dir": tmp_path, "file_pattern": "*.jpg"},
        models=[{"id": model_id, "adapter": {"type": "dummy"}} for model_id in ids],
        runner={"warmup_iterations": 0, "continue_on_error": True},
        comparison=comparison,
    )


def _prediction(
    image_id,
    elapsed=100,
    *,
    failed=False,
    schema=DUMMY_5_SCHEMA,
    poses=(),
    failure_kind=PredictionFailureKind.ADAPTER_EXECUTION,
):
    return PredictionResult(
        ImageInput(image_id, Path(f"{image_id}.jpg")),
        None if failed else AdapterOutput(schema, poses),
        PredictionTiming(image_id, elapsed),
        "adapter failed" if failed else None,
        failure_kind if failed else None,
    )


def _pose(x=0.5):
    return PosePrediction(tuple(Keypoint(name, x, 0.5) for name in DUMMY_5_SCHEMA.keypoint_names))


def _run(*predictions):
    return RunResult(WarmupResult(0, 0), tuple(predictions), 1000)


def _install_runners(monkeypatch, results, events=None):
    runners = {}
    for model_id, result in results.items():

        def run(model_id=model_id, result=result):
            if events is not None:
                events.append(("run", model_id))
            if isinstance(result, Exception):
                raise result
            return result

        runners[model_id] = Mock(spec=Runner, run=Mock(side_effect=run))
    factory = Mock(side_effect=lambda model, **kwargs: runners[model.id])
    monkeypatch.setattr(session_module, "create_model_runner", factory)
    return factory, runners


def test_runs_reference_once_then_candidates_in_config_order_and_computes_each_metric(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    runs = {
        "production": _run(_prediction("b", 100), _prediction("a", 200)),
        "candidate-z": _run(_prediction("a", 300), _prediction("b", 500)),
        "candidate-a": _run(_prediction("b", 900), _prediction("a", 1100)),
    }
    events = []
    factory, runners = _install_runners(monkeypatch, runs, events)
    engine = MetricsEngine()
    compare_spy = Mock(wraps=QualityComparator().compare)
    monkeypatch.setattr(session_module, "QualityComparator", lambda: Mock(compare=compare_spy))

    def compute(run):
        model_id = next(model_id for model_id, expected in runs.items() if run is expected)
        events.append(("metrics", model_id))
        return engine.compute(run)

    compute_spy = Mock(side_effect=compute)
    monkeypatch.setattr(session_module, "MetricsEngine", lambda: Mock(compute=compute_spy))

    result = EvaluationSession(config).run()

    assert events == [
        ("run", "production"),
        ("metrics", "production"),
        ("run", "candidate-z"),
        ("metrics", "candidate-z"),
        ("run", "candidate-a"),
        ("metrics", "candidate-a"),
    ]
    for runner in runners.values():
        runner.run.assert_called_once_with()
    assert compute_spy.call_count == 3
    assert compare_spy.call_count == 4
    for index, model_id in enumerate(("candidate-z", "candidate-a")):
        for offset, image_id in enumerate(("b", "a")):
            call = compare_spy.call_args_list[index * 2 + offset]
            assert call.args[0] == image_id
            assert call.args[1] is runs["production"].predictions[offset].output
            candidate_prediction = next(
                prediction
                for prediction in runs[model_id].predictions
                if prediction.image.image_id == image_id
            )
            assert call.args[2] is candidate_prediction.output
            assert call.args[4] is config.comparison
    calls = factory.call_args_list
    assert [call.args[0].id for call in calls] == [model.id for model in config.models]
    source = calls[0].kwargs["data_source"]
    assert isinstance(source, FileDataSource)
    assert source.input_dir == tmp_path
    assert all(call.kwargs["data_source"] is source for call in calls)
    assert all(call.kwargs["runner_config"] is config.runner for call in calls)
    assert result.reference.role == "reference"
    assert result.reference.run_result is runs["production"]
    assert result.reference.metrics.latency.mean_ns == 150
    assert [candidate.model.model_id for candidate in result.candidates] == [
        "candidate-z",
        "candidate-a",
    ]
    assert [candidate.model.metrics.latency.mean_ns for candidate in result.candidates] == [
        400,
        1000,
    ]
    for candidate in result.candidates:
        assert candidate.model.role == "candidate"
        assert candidate.model.run_result is runs[candidate.model.model_id]
        assert [image.image_id for image in candidate.image_comparisons] == ["b", "a"]
        assert candidate.schema_resolution.compared_keypoints == DUMMY_5_SCHEMA.keypoint_names
        assert candidate.quality_metrics.compared_image_count == 2
        assert candidate.quality_metrics.comparison_coverage == 1.0


def test_mixed_failures_preserve_reference_successes_and_count_candidate_misses(
    tmp_path, monkeypatch
):
    runs = {
        "reference": _run(
            _prediction("b", poses=(_pose(),)),
            _prediction("a", failed=True),
            _prediction("c", poses=(_pose(), _pose())),
            _prediction("d", failed=True),
        ),
        "candidate": _run(
            _prediction("c", failed=True),
            _prediction("d", failed=True),
            _prediction("a", poses=(_pose(), _pose(), _pose())),
            _prediction("b", poses=(_pose(),)),
        ),
    }
    _install_runners(monkeypatch, runs)
    result = EvaluationSession(_config(tmp_path, ("reference", "candidate"))).run()
    candidate = result.candidates[0]

    assert [image.image_id for image in candidate.image_comparisons] == ["b", "c"]
    assert result.reference.metrics.errors.failed_predictions == 2
    assert candidate.model.metrics.errors.failed_predictions == 2
    assert candidate.model.metrics.latency.sample_count == 2
    assert candidate.quality_metrics.comparison_coverage == 0.5
    assert candidate.quality_metrics.compared_image_count == 2
    assert candidate.quality_metrics.matched_pose_count == 1
    assert candidate.quality_metrics.unmatched_reference_pose_count == 2
    assert candidate.quality_metrics.unmatched_candidate_pose_count == 0
    assert candidate.quality_metrics.reference_pose_miss_rate == pytest.approx(2 / 3)
    failed_image = candidate.image_comparisons[1]
    assert failed_image.pose_matches == ()
    assert failed_image.unmatched_reference_pose_indices == (0, 1)
    assert failed_image.unmatched_candidate_pose_indices == ()
    failed_prediction = candidate.model.run_result.predictions[0]
    assert failed_prediction.output is None
    assert failed_prediction.error == "adapter failed"


def test_disjoint_successful_images_still_count_reference_success_for_coverage(
    tmp_path, monkeypatch
):
    _install_runners(
        monkeypatch,
        {
            "reference": _run(_prediction("a"), _prediction("b", failed=True)),
            "candidate": _run(_prediction("b"), _prediction("a", failed=True)),
        },
    )
    result = EvaluationSession(_config(tmp_path, ("reference", "candidate"))).run()
    candidate = result.candidates[0]
    assert [image.image_id for image in candidate.image_comparisons] == ["a"]
    assert candidate.quality_metrics.comparison_coverage == 0.5
    assert candidate.quality_metrics.mean_keypoint_deviation is None


@pytest.mark.parametrize("failure_kind", list(PredictionFailureKind))
def test_candidate_crash_cannot_disappear_from_pose_miss_statistics(
    tmp_path, monkeypatch, failure_kind
):
    reference = _run(
        _prediction("easy", poses=(_pose(),)),
        _prediction("hard", poses=(_pose(), _pose())),
    )
    baseline = _run(
        _prediction("hard", poses=(_pose(0.6), _pose(0.6))),
        _prediction("easy", poses=(_pose(),)),
    )
    crashed_prediction = _prediction("hard", failed=True, failure_kind=failure_kind)
    crashed = _run(crashed_prediction, _prediction("easy", poses=(_pose(),)))
    _install_runners(
        monkeypatch,
        {
            "reference": reference,
            "baseline": baseline,
            "crashed": crashed,
        },
    )

    result = EvaluationSession(_config(tmp_path, ("reference", "baseline", "crashed"))).run()
    good, bad = result.candidates
    assert good.quality_metrics.reference_pose_miss_rate == 0.0
    assert bad.quality_metrics.reference_pose_miss_rate == pytest.approx(2 / 3)
    assert bad.quality_metrics.unmatched_reference_pose_count == 2
    assert (
        bad.quality_metrics.compared_image_count == good.quality_metrics.compared_image_count == 2
    )
    assert (
        bad.quality_metrics.comparison_coverage == good.quality_metrics.comparison_coverage == 1.0
    )
    assert bad.model.metrics.errors.error_rate == 0.5
    assert bad.model.run_result.predictions[0] is crashed_prediction
    assert crashed_prediction.failure_kind is failure_kind
    # Deviation is conditional on actual matches: no fabricated sample penalizes a crash.
    assert good.quality_metrics.mean_keypoint_deviation > 0.0
    assert bad.quality_metrics.mean_keypoint_deviation == 0.0
    assert bad.quality_metrics.keypoint_sample_count == 5
    assert bad.quality_metrics.candidate_missing_keypoint_rate == 0.0


def test_candidate_failure_on_empty_reference_output_counts_for_coverage(tmp_path, monkeypatch):
    _install_runners(
        monkeypatch,
        {
            "reference": _run(_prediction("empty"), _prediction("a", failed=True)),
            "candidate": _run(_prediction("a"), _prediction("empty", failed=True)),
        },
    )
    result = EvaluationSession(_config(tmp_path, ("reference", "candidate"))).run()
    metrics = result.candidates[0].quality_metrics
    assert metrics.comparison_coverage == 0.5
    assert metrics.unmatched_reference_pose_count == 0
    assert metrics.reference_pose_miss_rate is None


@pytest.mark.parametrize("candidate_ids", [(), ("a",), ("a", "b", "extra"), ("a", "extra")])
def test_missing_or_unexpected_image_ids_raise_consistency_error_and_stop_session(
    tmp_path, monkeypatch, candidate_ids
):
    _, runners = _install_runners(
        monkeypatch,
        {
            "reference": _run(_prediction("a"), _prediction("b", failed=True)),
            "candidate": _run(*(_prediction(name) for name in candidate_ids)),
            "later": _run(_prediction("a"), _prediction("b")),
        },
    )
    with pytest.raises(EvaluationConsistencyError) as caught:
        EvaluationSession(_config(tmp_path, ("reference", "candidate", "later"))).run()
    message = str(caught.value)
    assert "'candidate' image IDs differ from reference 'reference'" in message
    assert f"missing image IDs: {sorted({'a', 'b'} - set(candidate_ids))!r}" in message
    assert f"unexpected image IDs: {sorted(set(candidate_ids) - {'a', 'b'})!r}" in message
    runners["later"].run.assert_not_called()


def test_comparison_results_independent_of_candidate_tuple_order(tmp_path, monkeypatch):
    reference = _run(
        _prediction("z", poses=(_pose(0.6),)),
        _prediction("a", poses=(_pose(),)),
        _prediction("b", poses=(_pose(), _pose())),
    )
    candidate = (
        _prediction("a", poses=(_pose(),)),
        _prediction("b", failed=True),
        _prediction("z", poses=(_pose(0.6),)),
    )
    _install_runners(
        monkeypatch,
        {
            "reference": reference,
            "first": _run(*candidate),
            "reordered": _run(*reversed(candidate)),
        },
    )
    result = EvaluationSession(_config(tmp_path, ("reference", "first", "reordered"))).run()
    first, reordered = result.candidates
    assert first.image_comparisons == reordered.image_comparisons
    assert first.quality_metrics == reordered.quality_metrics
    assert [image.image_id for image in first.image_comparisons] == ["z", "a", "b"]
    assert first.quality_metrics.matched_pose_count == 2
    assert first.quality_metrics.mean_keypoint_deviation == 0.0


@pytest.mark.parametrize("model_id", ["reference", "candidate"])
@pytest.mark.parametrize(
    "bad_run, message",
    [
        (_run(_prediction("a", failed=True)), "no successful output"),
        (
            _run(_prediction("a"), _prediction("b", schema=KeypointSchema("other", ("nose",)))),
            "changes schema",
        ),
        (_run(_prediction("a"), _prediction("a", failed=True)), "duplicate image ID"),
    ],
)
def test_rejects_unresolvable_schema_and_ambiguous_image_ids(
    tmp_path, monkeypatch, model_id, bad_run, message
):
    image_ids = dict.fromkeys(prediction.image.image_id for prediction in bad_run.predictions)
    runs = {
        name: _run(*(_prediction(image_id) for image_id in image_ids))
        for name in ("reference", "candidate")
    }
    runs[model_id] = bad_run
    _install_runners(monkeypatch, runs)
    with pytest.raises(ValueError, match=f"{model_id!r}.*{message}"):
        EvaluationSession(_config(tmp_path, ("reference", "candidate"))).run()


def test_empty_reference_cannot_resolve_schema(tmp_path, monkeypatch):
    _install_runners(monkeypatch, {"reference": _run()})
    with pytest.raises(ValueError, match="'reference'.*no successful output"):
        EvaluationSession(_config(tmp_path, ("reference",))).run()


@pytest.mark.parametrize("mode", ["strict", "common"])
def test_schema_resolution_uses_comparison_config(tmp_path, monkeypatch, mode):
    schema = KeypointSchema("subset", ("nose", "left_wrist", "right_wrist"))
    _install_runners(
        monkeypatch,
        {
            "reference": _run(_prediction("a")),
            "candidate": _run(_prediction("a", schema=schema)),
        },
    )
    session = EvaluationSession(
        _config(tmp_path, ("reference", "candidate"), schema_mode=mode, min_common_keypoints=3)
    )
    if mode == "strict":
        with pytest.raises(ValueError, match="Strict schema comparison"):
            session.run()
    else:
        resolution = session.run().candidates[0].schema_resolution
        assert resolution.compared_keypoints == schema.keypoint_names
        assert resolution.reference_only_keypoints == ("left_ankle", "right_ankle")


@pytest.mark.parametrize("failed_model", ["production", "candidate-z"])
def test_fail_fast_error_stops_later_models_without_retry(tmp_path, monkeypatch, failed_model):
    error = RunnerExecutionError("prediction failed")
    runs = {model.id: _run(_prediction("a")) for model in _config(tmp_path).models}
    runs[failed_model] = error
    _, runners = _install_runners(monkeypatch, runs)
    with pytest.raises(RunnerExecutionError) as caught:
        EvaluationSession(_config(tmp_path)).run()
    assert caught.value is error
    runners[failed_model].run.assert_called_once_with()
    runners["candidate-a"].run.assert_not_called()
    if failed_model == "production":
        runners["candidate-z"].run.assert_not_called()


def test_reference_only_session_runs_once(tmp_path, monkeypatch):
    _, runners = _install_runners(monkeypatch, {"reference": _run(_prediction("a"))})
    result = EvaluationSession(_config(tmp_path, ("reference",))).run()
    runners["reference"].run.assert_called_once_with()
    assert result.reference.model_id == "reference"
    assert result.candidates == ()
