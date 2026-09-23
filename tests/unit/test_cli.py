from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput
from pose_deploy_gate.cli import run
from pose_deploy_gate.runner.result import (
    PredictionResult,
    PredictionTiming,
    RunResult,
    WarmupResult,
)


def _prediction(
    image_id: str,
    elapsed_ns: int,
    *,
    error: str | None = None,
) -> PredictionResult:
    return PredictionResult(
        image=ImageInput(image_id=image_id, path=Path(f"/inputs/{image_id}.jpg")),
        output=None if error else AdapterOutput(poses=()),
        timing=PredictionTiming(image_id=image_id, elapsed_ns=elapsed_ns),
        error=error,
    )


def _run_result(*predictions: PredictionResult) -> RunResult:
    return RunResult(
        warmup=WarmupResult(iterations=3, total_time_ns=500_000),
        predictions=predictions,
        total_time_ns=20_000_000,
    )


def _run_cli(
    run_result: RunResult,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> str:
    class FakeRunner:
        def run(self) -> RunResult:
            return run_result

    config = SimpleNamespace(
        run=SimpleNamespace(name="test-run"),
        data=SimpleNamespace(input_dir=tmp_path),
        adapter=SimpleNamespace(type="dummy"),
        output=SimpleNamespace(dir=tmp_path / "output"),
        gates=SimpleNamespace(enabled=True),
    )
    monkeypatch.setattr("pose_deploy_gate.cli.load_config", lambda path: config)
    monkeypatch.setattr("pose_deploy_gate.cli.create_runner", lambda config: FakeRunner())
    args = Namespace(
        config=tmp_path / "config.yaml",
        input=None,
        strict=False,
        list_inputs=False,
    )

    assert run(args) == 0
    return capsys.readouterr().out


def _percentile_run_result() -> RunResult:
    return _run_result(*(_prediction(str(value), value * 1_000_000) for value in range(1, 101)))


def test_cli_prints_latency_metrics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = _run_cli(
        _run_result(
            _prediction("one", 1_000_000),
            _prediction("two", 2_000_000),
            _prediction("three", 3_000_000),
        ),
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert "Latency:" in output
    assert "  Samples: 3" in output
    assert "  Min: 1.000 ms" in output
    assert "  Mean: 2.000 ms" in output
    assert "  Max: 3.000 ms" in output


def test_cli_prints_p50_latency(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = _run_cli(_percentile_run_result(), monkeypatch, capsys, tmp_path)

    assert "  P50: 50.000 ms" in output


def test_cli_prints_p95_latency(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = _run_cli(_percentile_run_result(), monkeypatch, capsys, tmp_path)

    assert "  P95: 95.000 ms" in output


def test_cli_prints_p99_latency(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = _run_cli(_percentile_run_result(), monkeypatch, capsys, tmp_path)

    assert "  P99: 99.000 ms" in output


def test_cli_prints_error_rate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = _run_cli(
        _run_result(
            _prediction("success-1", 1_000_000),
            _prediction("success-2", 2_000_000),
            _prediction("success-3", 3_000_000),
            _prediction("failure", 4_000_000, error="adapter failed"),
        ),
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert "Reliability:" in output
    assert "  Successful predictions: 3" in output
    assert "  Failed predictions: 1" in output
    assert "  Error rate: 25.00%" in output


def test_cli_prints_zero_error_rate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = _run_cli(
        _run_result(_prediction("one", 1_000_000), _prediction("two", 2_000_000)),
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert "  Error rate: 0.00%" in output


def test_cli_prints_na_for_missing_latency_samples(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = _run_cli(
        _run_result(
            _prediction("failure-1", 1_000_000, error="adapter failed"),
            _prediction("failure-2", 2_000_000, error="timeout"),
        ),
        monkeypatch,
        capsys,
        tmp_path,
    )

    assert "  Samples: 0" in output
    for label in ("Min", "Mean", "P50", "P95", "P99", "Max"):
        assert f"  {label}: n/a" in output
        assert f"  {label}: 0.000 ms" not in output
