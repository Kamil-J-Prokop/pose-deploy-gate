from dataclasses import replace
from unittest.mock import Mock

from pose_deploy_gate.adapters import DummyAdapter
from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.adapters.exceptions import AdapterExecutionError
from pose_deploy_gate.config import load_config
from pose_deploy_gate.config.models import AppConfig
from pose_deploy_gate.evaluation import EvaluationSession
from pose_deploy_gate.evaluation import session as session_module
from pose_deploy_gate.runner import PredictionFailureKind, create_model_runner


def test_config_drives_real_runners_and_quality_comparisons(tmp_path, monkeypatch):
    for image in ("a.jpg", "b.jpg"):
        (tmp_path / image).touch()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "version: 1\n"
        "data:\n"
        f"  input_dir: {tmp_path}\n"
        "  file_pattern: '*.jpg'\n"
        "runner:\n"
        "  warmup_iterations: 0\n"
        "models:\n"
        "  - id: production\n"
        "    adapter: {type: dummy}\n"
        "  - id: candidate-z\n"
        "    adapter: {type: dummy, params: {pose_confidence: 0.8}}\n"
        "  - id: candidate-a\n"
        "    adapter: {type: dummy, params: {pose_confidence: 0.6}}\n"
    )
    runners = []

    def create_runner(*args, **kwargs):
        runner = create_model_runner(*args, **kwargs)
        runners.append(runner)
        return Mock(wraps=runner)

    factory = Mock(side_effect=create_runner)
    monkeypatch.setattr(session_module, "create_model_runner", factory)
    session = EvaluationSession(load_config(config_path))

    result = session.run()

    assert len(runners) == 3
    assert all(runner.data_source is runners[0].data_source for runner in runners)
    assert len({id(runner.adapter) for runner in runners}) == 3
    assert [runner.adapter.pose_confidence for runner in runners] == [1.0, 0.8, 0.6]
    assert result.reference.metrics.errors.total_attempts == 2
    assert [candidate.model.model_id for candidate in result.candidates] == [
        "candidate-z",
        "candidate-a",
    ]
    for candidate in result.candidates:
        assert candidate.model.metrics.latency.sample_count == 2
        assert candidate.quality_metrics.comparison_coverage == 1.0
        assert candidate.quality_metrics.matched_pose_count == 2
        assert candidate.quality_metrics.mean_keypoint_deviation == 0.0
        assert [image.image_id for image in candidate.image_comparisons] == ["a", "b"]

    # A second invocation starts a new evaluation rather than reusing old runs.
    second = session.run()
    assert len(runners) == 6
    assert runners[3].data_source is not runners[0].data_source
    assert second.reference.run_result is not result.reference.run_result


class FailingAdapter(PoseAdapter):
    def __init__(self, failures):
        self.failures = failures

    @property
    def name(self):
        return "failing-dummy"

    def predict(self, image):
        failure = self.failures.get(image.image_id)
        if failure == "crash":
            raise AdapterExecutionError("scripted failure")
        return DummyAdapter(keypoint_confidence=2.0 if failure == "invalid" else 1.0).predict(image)


def test_real_prediction_failures_preserve_misses_and_reference_coverage(tmp_path, monkeypatch):
    for image_id in ("a", "b", "c", "d"):
        (tmp_path / f"{image_id}.jpg").touch()
    failures = {
        "reference": {"b": "crash", "c": "invalid"},
        "candidate": {"a": "invalid", "c": "crash"},
        "healthy": {},
    }

    def create_runner(model, **kwargs):
        return replace(
            create_model_runner(model, **kwargs), adapter=FailingAdapter(failures[model.id])
        )

    monkeypatch.setattr(session_module, "create_model_runner", create_runner)
    config = AppConfig(
        version=1,
        data={"input_dir": tmp_path, "file_pattern": "*.jpg"},
        models=[{"id": name, "adapter": {"type": "dummy"}} for name in failures],
        runner={"warmup_iterations": 0, "continue_on_error": True},
    )

    result = EvaluationSession(config).run()

    assert result.reference.metrics.errors.error_rate == 0.5
    candidate, healthy = result.candidates
    assert candidate.model.metrics.errors.error_rate == 0.5
    assert candidate.model.metrics.latency.sample_count == 2
    failed = candidate.model.run_result.predictions[0]
    assert failed.output is None
    assert failed.failure_kind is PredictionFailureKind.OUTPUT_VALIDATION
    assert (
        candidate.model.run_result.predictions[2].failure_kind
        is PredictionFailureKind.ADAPTER_EXECUTION
    )
    assert [image.image_id for image in candidate.image_comparisons] == ["a", "d"]
    assert candidate.image_comparisons[0].unmatched_reference_pose_indices == (0,)
    assert candidate.quality_metrics.matched_pose_count == 1
    assert candidate.quality_metrics.unmatched_reference_pose_count == 1
    assert candidate.quality_metrics.reference_pose_miss_rate == 0.5
    assert (
        candidate.quality_metrics.comparison_coverage
        == healthy.quality_metrics.comparison_coverage
        == 0.5
    )
    assert healthy.quality_metrics.reference_pose_miss_rate == 0.0
    assert candidate.quality_metrics.mean_keypoint_deviation == 0.0
