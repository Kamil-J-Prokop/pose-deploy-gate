from pathlib import Path

from pose_deploy_gate.adapters import DummyAdapter
from pose_deploy_gate.data import FileDataSource
from pose_deploy_gate.metrics import MetricsEngine
from pose_deploy_gate.runner import Runner


def test_runner_results_produce_consistent_metrics(image_fixtures_dir: Path) -> None:
    fixture_count = len(tuple(image_fixtures_dir.rglob("*.jpg")))
    runner = Runner(
        adapter=DummyAdapter(),
        data_source=FileDataSource(
            input_dir=image_fixtures_dir,
            file_pattern="*.jpg",
            recursive=True,
        ),
    )

    run_result = runner.run()
    metrics = MetricsEngine().compute(run_result)

    assert metrics.errors.total_attempts == fixture_count
    assert metrics.errors.successful_predictions == fixture_count
    assert metrics.errors.failed_predictions == 0
    assert metrics.errors.error_rate == 0.0
    assert metrics.latency.sample_count == fixture_count

    assert metrics.latency.min_ns is not None
    assert metrics.latency.p50_ns is not None
    assert metrics.latency.p95_ns is not None
    assert metrics.latency.p99_ns is not None
    assert metrics.latency.max_ns is not None
    assert (
        0
        <= metrics.latency.min_ns
        <= metrics.latency.p50_ns
        <= metrics.latency.p95_ns
        <= metrics.latency.p99_ns
        <= metrics.latency.max_ns
    )
