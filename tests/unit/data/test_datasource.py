from pathlib import Path

from pose_deploy_gate.adapters.types import ImageInput
from pose_deploy_gate.config.models import DataConfig
from pose_deploy_gate.data.datasource import FileDataSource


def test_file_data_source_iter_images_returns_image_inputs(tmp_path: Path) -> None:
    first_file = tmp_path / "frame_001.jpg"
    second_file = tmp_path / "frame_002.jpg"
    first_file.write_text("a", encoding="utf-8")
    second_file.write_text("b", encoding="utf-8")

    data_source = FileDataSource(input_dir=tmp_path)

    images = tuple(data_source.iter_images())

    assert images == (
        ImageInput(image_id="frame_001", path=first_file),
        ImageInput(image_id="frame_002", path=second_file),
    )


def test_file_data_source_generates_image_id_from_relative_path_without_suffix(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "frame.jpg"
    file_path.write_text("frame", encoding="utf-8")

    data_source = FileDataSource(input_dir=tmp_path)

    images = tuple(data_source.iter_images())

    assert images[0].image_id == "frame"


def test_file_data_source_preserves_nested_relative_path_in_image_id(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    file_path = nested_dir / "frame.jpg"
    file_path.write_text("frame", encoding="utf-8")

    data_source = FileDataSource(input_dir=tmp_path, recursive=True)

    images = tuple(data_source.iter_images())

    assert images[0].image_id == "nested/frame"


def test_file_data_source_uses_file_pattern(tmp_path: Path) -> None:
    jpg_file = tmp_path / "frame.jpg"
    txt_file = tmp_path / "notes.txt"
    jpg_file.write_text("jpg", encoding="utf-8")
    txt_file.write_text("txt", encoding="utf-8")

    data_source = FileDataSource(input_dir=tmp_path, file_pattern="*.jpg")

    images = tuple(data_source.iter_images())

    assert images == (ImageInput(image_id="frame", path=jpg_file),)


def test_file_data_source_uses_recursive_flag(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    root_file = tmp_path / "root.jpg"
    nested_file = nested_dir / "nested.jpg"
    root_file.write_text("root", encoding="utf-8")
    nested_file.write_text("nested", encoding="utf-8")

    data_source = FileDataSource(input_dir=tmp_path, recursive=False)

    images = tuple(data_source.iter_images())

    assert images == (ImageInput(image_id="root", path=root_file),)


def test_file_data_source_from_config(tmp_path: Path) -> None:
    config = DataConfig(
        input_dir=tmp_path,
        file_pattern="*.png",
        recursive=True,
    )

    data_source = FileDataSource.from_config(config)

    assert data_source == FileDataSource(
        input_dir=tmp_path,
        file_pattern="*.png",
        recursive=True,
    )


def test_file_data_source_iter_images_is_repeatable(tmp_path: Path) -> None:
    file_path = tmp_path / "frame.jpg"
    file_path.write_text("frame", encoding="utf-8")

    data_source = FileDataSource(input_dir=tmp_path)

    first = tuple(data_source.iter_images())
    second = tuple(data_source.iter_images())

    assert first == second
