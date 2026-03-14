from argparse import Namespace
from pathlib import Path

from pose_deploy_gate.cli import run


def test_run_without_input_no_strict() -> None:
    args = Namespace(
        input=None,
        strict=False,
    )
    assert run(args) == 0


def test_run_without_input_strict() -> None:
    args = Namespace(
        input=None,
        strict=True,
    )
    assert run(args) == 2


def test_run_with_existing_directory(tmp_path: Path) -> None:
    args = Namespace(
        input=tmp_path,
        strict=False,
    )
    assert run(args) == 0


def test_run_with_nonexistent_directory(tmp_path: Path) -> None:
    wrong_path = tmp_path / "wrong_path"
    args = Namespace(
        input=wrong_path,
        strict=False,
    )
    assert run(args) == 1
