from argparse import Namespace
from pathlib import Path

from pose_deploy_gate.cli import run


def test_run_without_input_no_strict() -> None:
    args = Namespace(
        config=None,
        input=None,
        strict=False,
    )
    assert run(args) == 0


def test_run_without_input_strict() -> None:
    args = Namespace(
        config=None,
        input=None,
        strict=True,
    )
    assert run(args) == 2


def test_run_with_existing_directory(tmp_path: Path) -> None:
    args = Namespace(
        config=None,
        input=tmp_path,
        strict=False,
    )
    assert run(args) == 0


def test_run_with_nonexistent_directory(tmp_path: Path) -> None:
    wrong_path = tmp_path / "wrong_path"
    args = Namespace(
        config=None,
        input=wrong_path,
        strict=False,
    )
    assert run(args) == 1


def test_run_with_valid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
version: 1

data:
  input_dir: "{tmp_path}"

adapter:
  type: "dummy"
""",
        encoding="utf-8",
    )
    args = Namespace(
        config=config_path,
        input=None,
        strict=False,
    )

    assert run(args) == 0


def test_run_with_invalid_config_path(tmp_path: Path) -> None:
    args = Namespace(
        config=tmp_path / "missing.yaml",
        input=None,
        strict=False,
    )

    assert run(args) == 2
