from pathlib import Path

import pytest

from pose_deploy_gate.data.exceptions import InputDirectoryError, NoInputFilesError
from pose_deploy_gate.data.file_discovery import discover_files


def test_discover_files_returns_matching_files(image_fixtures_dir: Path) -> None:
    files = discover_files(image_fixtures_dir)

    assert files == (
        image_fixtures_dir / "a.jpg",
        image_fixtures_dir / "b.jpg",
    )


def test_discover_files_ignores_directories(image_fixtures_dir: Path) -> None:
    files = discover_files(image_fixtures_dir)

    assert files == (
        image_fixtures_dir / "a.jpg",
        image_fixtures_dir / "b.jpg",
    )


def test_discover_files_is_sorted_by_relative_path(tmp_path: Path) -> None:
    z_dir = tmp_path / "z_dir"
    a_dir = tmp_path / "a_dir"
    z_dir.mkdir()
    a_dir.mkdir()

    z_file = z_dir / "frame.txt"
    root_file = tmp_path / "m_frame.txt"
    a_file = a_dir / "frame.txt"
    z_file.write_text("z", encoding="utf-8")
    root_file.write_text("m", encoding="utf-8")
    a_file.write_text("a", encoding="utf-8")

    files = discover_files(tmp_path, recursive=True)

    assert files == (a_file, root_file, z_file)


def test_discover_files_non_recursive_ignores_nested_files(
    image_fixtures_dir: Path,
) -> None:
    files = discover_files(image_fixtures_dir)

    assert files == (
        image_fixtures_dir / "a.jpg",
        image_fixtures_dir / "b.jpg",
    )


def test_discover_files_recursive_includes_nested_files(
    image_fixtures_dir: Path,
) -> None:
    files = discover_files(image_fixtures_dir, recursive=True)

    assert files == (
        image_fixtures_dir / "a.jpg",
        image_fixtures_dir / "b.jpg",
        image_fixtures_dir / "nested" / "c.jpg",
        image_fixtures_dir / "nested" / "d.png",
    )


def test_discover_files_respects_file_pattern(image_fixtures_dir: Path) -> None:
    files = discover_files(image_fixtures_dir, file_pattern="*.jpg", recursive=True)

    assert files == (
        image_fixtures_dir / "a.jpg",
        image_fixtures_dir / "b.jpg",
        image_fixtures_dir / "nested" / "c.jpg",
    )


def test_discover_files_raises_for_missing_input_dir(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(InputDirectoryError, match="does not exist"):
        discover_files(missing_dir)


def test_discover_files_raises_when_input_path_is_file(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(InputDirectoryError, match="not a directory"):
        discover_files(file_path)


def test_discover_files_raises_when_no_files_match(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()

    with pytest.raises(NoInputFilesError, match="No input files found"):
        discover_files(tmp_path)
