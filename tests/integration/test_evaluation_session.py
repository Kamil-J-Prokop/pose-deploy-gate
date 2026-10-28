from unittest.mock import Mock

from pose_deploy_gate.config import load_config
from pose_deploy_gate.evaluation import EvaluationSession
from pose_deploy_gate.evaluation import session as session_module
from pose_deploy_gate.runner import create_model_runner


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
