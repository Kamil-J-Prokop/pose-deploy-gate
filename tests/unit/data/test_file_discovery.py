from pathlib import Path

import pytest

from pose_deploy_gate.data.exceptions import InputDirectoryError, NoInputFilesError
from pose_deploy_gate.data.file_discovery import discover_files


def test_discover_files_returns_matching_files(tmp_path: Path) -> None:
    first_file = tmp_path / "frame_001.jpg"
    second_file = tmp_path / "frame_002.jpg"
    first_file.write_text("a", encoding="utf-8")
    second_file.write_text("b", encoding="utf-8")

    files = discover_files(tmp_path)

    assert files == (first_file, second_file)


def test_discover_files_ignores_directories(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    file_path = tmp_path / "frame.jpg"
    file_path.write_text("frame", encoding="utf-8")

    files = discover_files(tmp_path)

    assert files == (file_path,)


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


def test_discover_files_non_recursive_ignores_nested_files(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    root_file = tmp_path / "root.txt"
    nested_file = nested_dir / "nested.txt"
    root_file.write_text("root", encoding="utf-8")
    nested_file.write_text("nested", encoding="utf-8")

    files = discover_files(tmp_path)

    assert files == (root_file,)


def test_discover_files_recursive_includes_nested_files(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    root_file = tmp_path / "root.txt"
    nested_file = nested_dir / "nested.txt"
    root_file.write_text("root", encoding="utf-8")
    nested_file.write_text("nested", encoding="utf-8")

    files = discover_files(tmp_path, recursive=True)

    assert files == (nested_file, root_file)


def test_discover_files_respects_file_pattern(tmp_path: Path) -> None:
    jpg_file = tmp_path / "frame.jpg"
    txt_file = tmp_path / "notes.txt"
    jpg_file.write_text("jpg", encoding="utf-8")
    txt_file.write_text("txt", encoding="utf-8")

    files = discover_files(tmp_path, file_pattern="*.jpg")

    assert files == (jpg_file,)


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
