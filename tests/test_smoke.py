from argparse import Namespace
from pathlib import Path

from pose_deploy_gate.cli import run


def test_run_without_input_no_strict() -> None:
    args = Namespace(
        config=None,
        input=None,
        strict=False,
        list_inputs=False,
    )
    assert run(args) == 0


def test_run_without_input_strict() -> None:
    args = Namespace(
        config=None,
        input=None,
        strict=True,
        list_inputs=False,
    )
    assert run(args) == 2


def test_run_with_existing_directory(tmp_path: Path) -> None:
    args = Namespace(
        config=None,
        input=tmp_path,
        strict=False,
        list_inputs=False,
    )
    assert run(args) == 0


def test_run_with_nonexistent_directory(tmp_path: Path) -> None:
    wrong_path = tmp_path / "wrong_path"
    args = Namespace(
        config=None,
        input=wrong_path,
        strict=False,
        list_inputs=False,
    )
    assert run(args) == 1


def test_run_with_valid_config(tmp_path: Path, image_fixtures_dir: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
version: 1

data:
  input_dir: "{image_fixtures_dir}"

adapter:
  type: "dummy"
""",
        encoding="utf-8",
    )
    args = Namespace(
        config=config_path,
        input=None,
        strict=False,
        list_inputs=False,
    )

    assert run(args) == 0


def test_cli_config_path_initializes_dummy_adapter(
    tmp_path: Path, capsys, image_fixtures_dir: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
version: 1

data:
  input_dir: "{image_fixtures_dir}"

adapter:
  type: "dummy"
""",
        encoding="utf-8",
    )
    args = Namespace(
        config=config_path,
        input=None,
        strict=False,
        list_inputs=False,
    )

    assert run(args) == 0

    captured = capsys.readouterr()
    assert "Adapter: dummy" in captured.out
    assert "Adapter initialized: dummy" in captured.out


def test_cli_config_reports_discovered_input_count(
    tmp_path: Path, capsys, image_fixtures_dir: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
version: 1

data:
  input_dir: "{image_fixtures_dir}"

adapter:
  type: "dummy"
""",
        encoding="utf-8",
    )
    args = Namespace(
        config=config_path,
        input=None,
        strict=False,
        list_inputs=False,
    )

    assert run(args) == 0

    captured = capsys.readouterr()
    assert "Input files discovered: 2" in captured.out


def test_cli_config_list_inputs_prints_files_in_deterministic_order(
    tmp_path: Path, capsys, image_fixtures_dir: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
version: 1

data:
  input_dir: "{image_fixtures_dir}"
  recursive: true
  file_pattern: "*.jpg"

adapter:
  type: "dummy"
""",
        encoding="utf-8",
    )
    args = Namespace(
        config=config_path,
        input=None,
        strict=False,
        list_inputs=True,
    )

    assert run(args) == 0

    captured = capsys.readouterr()
    assert "Input files:" in captured.out
    assert "  001: a.jpg" in captured.out
    assert "  002: b.jpg" in captured.out
    assert "  003: nested/c.jpg" in captured.out


def test_cli_config_returns_error_when_no_files_match(tmp_path: Path, capsys) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
version: 1

data:
  input_dir: "{input_dir}"
  file_pattern: "*.jpg"

adapter:
  type: "dummy"
""",
        encoding="utf-8",
    )
    args = Namespace(
        config=config_path,
        input=None,
        strict=False,
        list_inputs=False,
    )

    assert run(args) == 2

    captured = capsys.readouterr()
    assert "ERROR: No input files found" in captured.out


def test_run_with_invalid_config_path(tmp_path: Path) -> None:
    args = Namespace(
        config=tmp_path / "missing.yaml",
        input=None,
        strict=False,
        list_inputs=False,
    )

    assert run(args) == 2
