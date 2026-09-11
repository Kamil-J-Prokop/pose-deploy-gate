import re
from argparse import Namespace
from pathlib import Path

import pytest

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
                "adapter:",
                '  type: "dummy"',
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
    assert "Adapter: dummy" in output
    assert "PoseDeployGate run completed." in output
    assert "Input files: 2" in output
    assert output.count("Config path:") == 1
    assert output.count("Run name:") == 1
    assert output.count("Adapter:") == 1


def test_cli_config_prints_timing_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    image_fixtures_dir: Path,
) -> None:
    args = _config_args(tmp_path, image_fixtures_dir)

    assert run(args) == 0

    output = capsys.readouterr().out
    assert "Warmup iterations: 3" in output
    assert "Successful predictions: 2" in output
    assert "Failed predictions: 0" in output
    for label in (
        "Average inference time:",
        "Total measured inference time:",
        "Total runner time:",
    ):
        assert re.search(rf"^{re.escape(label)} \d+\.\d{{3}} ms$", output, re.MULTILINE)


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
    assert "PoseDeployGate run completed." not in output


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
        "pose_deploy_gate.cli.create_runner",
        lambda config: FailingRunner(),
    )
    args = _config_args(tmp_path, image_fixtures_dir)

    assert run(args) == 2

    output = capsys.readouterr().out
    assert "ERROR: runner failed" in output
    assert "PoseDeployGate run completed." not in output


def test_run_with_invalid_config_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "missing.yaml"

    assert run(_args(config=config_path)) == 2

    output = capsys.readouterr().out
    assert output == f"ERROR: Config file does not exist: {config_path}\n"
    assert "PoseDeployGate run completed." not in output
