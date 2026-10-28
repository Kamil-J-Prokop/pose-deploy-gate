import re
from argparse import Namespace
from pathlib import Path

import pytest

from pose_deploy_gate.adapters import DummyAdapter
from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput, Keypoint, PosePrediction
from pose_deploy_gate.cli import run
from pose_deploy_gate.runner import RunnerExecutionError


def _args(
    *,
    config: Path | None = None,
    input_path: Path | None = None,
    strict: bool = False,
    list_inputs: bool = False,
) -> Namespace:
    return Namespace(
        config=config,
        input=input_path,
        strict=strict,
        list_inputs=list_inputs,
    )


def _config_args(
    tmp_path: Path,
    input_dir: Path,
    *,
    recursive: bool = False,
    file_pattern: str | None = None,
    list_inputs: bool = False,
) -> Namespace:
    data_options = []
    if recursive:
        data_options.append("  recursive: true")
    if file_pattern is not None:
        data_options.append(f'  file_pattern: "{file_pattern}"')

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "version: 1",
                "",
                "data:",
                f'  input_dir: "{input_dir}"',
                *data_options,
                "",
                "models:",
                "  - id: dummy",
                "    adapter:",
                '      type: "dummy"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return _args(config=config_path, list_inputs=list_inputs)


def test_run_without_input_no_strict(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(_args()) == 0

    assert capsys.readouterr().out.splitlines() == [
        "PoseDeployGate CLI is wired correctly.",
        "Warning: No --input provided. Skipping validation.",
    ]


def test_run_without_input_strict(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(_args(strict=True)) == 2

    assert capsys.readouterr().out == "ERROR: --input is required when --strict is set.\n"


def test_run_with_existing_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(_args(input_path=tmp_path)) == 0

    assert capsys.readouterr().out.splitlines() == [
        "PoseDeployGate input validation successful.",
        f"Resolved path: {tmp_path.resolve()}, Path type: (directory)",
    ]


def test_run_with_nonexistent_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wrong_path = tmp_path / "wrong_path"

    assert run(_args(input_path=wrong_path)) == 1

    assert capsys.readouterr().out == (
        f"ERROR: The specified input path '{wrong_path}' does not exist.\n"
    )


def test_cli_config_runs_runner_successfully(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    image_fixtures_dir: Path,
) -> None:
    args = _config_args(tmp_path, image_fixtures_dir)

    assert run(args) == 0

    output = capsys.readouterr().out
    assert "PoseDeployGate config validation successful." in output
    assert f"Config path: {args.config.resolve()}" in output
    assert "Run name: default-run" in output
    assert f"Input directory: {image_fixtures_dir.resolve()}" in output
    assert "Reference: dummy" in output
    assert "PoseDeployGate evaluation completed." in output
    assert "  Images: 2" in output
    assert "Candidate:" not in output
    assert output.count("Config path:") == 1
    assert output.count("Run name:") == 1
    assert output.count("Reference:") == 1


def test_cli_config_prints_timing_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    image_fixtures_dir: Path,
) -> None:
    args = _config_args(tmp_path, image_fixtures_dir)

    assert run(args) == 0

    output = capsys.readouterr().out
    assert "Successful predictions: 2" in output
    assert "Failed predictions: 0" in output
    assert "  Failure rate: 0.00%" in output
    assert re.search(r"^  P95 latency: \d+\.\d{3} ms$", output, re.MULTILINE)


def test_cli_config_list_inputs_prints_files_in_deterministic_order(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    image_fixtures_dir: Path,
) -> None:
    args = _config_args(
        tmp_path,
        image_fixtures_dir,
        recursive=True,
        file_pattern="*.jpg",
        list_inputs=True,
    )

    assert run(args) == 0

    output_lines = capsys.readouterr().out.splitlines()
    list_heading_index = output_lines.index("Input files:")
    assert output_lines[list_heading_index + 1 :] == [
        "  001: a.jpg",
        "  002: b.jpg",
        "  003: nested/c.jpg",
    ]


def test_cli_config_returns_error_when_no_files_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    args = _config_args(tmp_path, input_dir, file_pattern="*.jpg")

    assert run(args) == 2

    output = capsys.readouterr().out
    assert "ERROR: No input files found" in output
    assert "PoseDeployGate evaluation completed." not in output


def test_cli_config_returns_error_when_runner_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    image_fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRunner:
        def run(self) -> None:
            raise RunnerExecutionError("runner failed")

    monkeypatch.setattr(
        "pose_deploy_gate.evaluation.session.create_model_runner",
        lambda model, **kwargs: FailingRunner(),
    )
    args = _config_args(tmp_path, image_fixtures_dir)

    assert run(args) == 2

    output = capsys.readouterr().out
    assert "ERROR: runner failed" in output
    assert "PoseDeployGate evaluation completed." not in output


def test_run_with_invalid_config_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "missing.yaml"

    assert run(_args(config=config_path)) == 2

    output = capsys.readouterr().out
    assert output == f"ERROR: Config file does not exist: {config_path}\n"
    assert "PoseDeployGate evaluation completed." not in output


def test_cli_multiple_models_prints_all_candidate_comparisons(tmp_path, capsys):
    (tmp_path / "a.jpg").touch()
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        f'version: 1\ndata:\n  input_dir: "{tmp_path}"\n  file_pattern: "*.jpg"\nmodels:\n'
        "  - id: production\n    adapter:\n      type: dummy\n"
        "  - id: candidate-z\n    adapter:\n      type: dummy\n"
        "  - id: candidate-a\n    adapter:\n      type: dummy\n",
        encoding="utf-8",
    )
    assert run(_args(config=config_path, list_inputs=True)) == 0
    output = capsys.readouterr().out
    assert "PoseDeployGate evaluation completed." in output
    assert (
        output.index("Reference: production")
        < output.index("Candidate: candidate-z")
        < output.index("Candidate: candidate-a")
    )
    assert output.count("    Comparison coverage: 100.00%") == 2
    assert output.count("    Mean keypoint deviation: 0.00%") == 2
    assert output.count("    Matched poses: 1") == 2
    assert output.count("    Compared keypoints: 5/5 reference, 5/5 candidate") == 2
    assert output.count("Input files:") == 1
    assert output.endswith("  001: a.jpg\n")


def test_cli_common_schema_comparison_end_to_end(tmp_path, capsys, monkeypatch):
    (tmp_path / "a.jpg").touch()
    reference_schema = KeypointSchema("reference_4", ("nose", "left", "right", "ref_only"))
    candidate_schema = KeypointSchema(
        "candidate_5", ("nose", "left", "right", "extra_1", "extra_2")
    )

    def predict(adapter, image):
        schema = reference_schema if adapter.pose_confidence == 1.0 else candidate_schema
        x = 0.5 if adapter.pose_confidence == 1.0 else 0.51
        return AdapterOutput(
            schema,
            (PosePrediction(tuple(Keypoint(name, x, 0.5) for name in schema.keypoint_names)),),
        )

    monkeypatch.setattr(DummyAdapter, "predict", predict)
    config_path = tmp_path / "common.yaml"
    config_path.write_text(
        f'version: 1\ndata:\n  input_dir: "{tmp_path}"\n  file_pattern: "*.jpg"\n'
        "models:\n"
        "  - id: production\n    adapter: {type: dummy}\n"
        "  - id: custom\n    adapter: {type: dummy, params: {pose_confidence: 0.8}}\n"
        "comparison:\n  schema_mode: common\n  min_common_keypoints: 3\n",
        encoding="utf-8",
    )
    assert run(_args(config=config_path)) == 0
    output = capsys.readouterr().out
    assert "    Reference: reference_4" in output
    assert "    Candidate: candidate_5" in output
    assert "    Compared keypoints: 3/4 reference, 3/5 candidate" in output
    assert "    Reference-only keypoints excluded: 1" in output
    assert "    Candidate-only keypoints excluded: 2" in output
    assert "    Comparison coverage: 100.00%" in output
    assert "    Mean keypoint deviation: 0.71%" in output
    assert "    P95 keypoint deviation: 0.71%" in output
    assert "    Missing keypoint rate: 0.00%" in output
    assert "nose" not in output
