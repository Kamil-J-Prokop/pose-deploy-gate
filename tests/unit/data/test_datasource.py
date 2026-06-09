from pathlib import Path

from pose_deploy_gate.adapters.types import ImageInput
from pose_deploy_gate.config.models import DataConfig
from pose_deploy_gate.data.datasource import FileDataSource


def test_file_data_source_iter_images_returns_image_inputs(
    image_fixtures_dir: Path,
) -> None:
    data_source = FileDataSource(input_dir=image_fixtures_dir)

    images = tuple(data_source.iter_images())

    assert images == (
        ImageInput(image_id="a", path=image_fixtures_dir / "a.jpg"),
        ImageInput(image_id="b", path=image_fixtures_dir / "b.jpg"),
    )


def test_file_data_source_generates_image_id_from_relative_path_without_suffix(
    image_fixtures_dir: Path,
) -> None:
    data_source = FileDataSource(input_dir=image_fixtures_dir)

    images = tuple(data_source.iter_images())

    assert images[0].image_id == "a"


def test_file_data_source_preserves_nested_relative_path_in_image_id(
    image_fixtures_dir: Path,
) -> None:
    data_source = FileDataSource(input_dir=image_fixtures_dir, recursive=True)

    images = tuple(data_source.iter_images())

    assert images[2].image_id == "nested/c"


def test_file_data_source_uses_file_pattern(image_fixtures_dir: Path) -> None:
    data_source = FileDataSource(
        input_dir=image_fixtures_dir,
        file_pattern="*.jpg",
        recursive=True,
    )

    images = tuple(data_source.iter_images())

    assert images == (
        ImageInput(image_id="a", path=image_fixtures_dir / "a.jpg"),
        ImageInput(image_id="b", path=image_fixtures_dir / "b.jpg"),
        ImageInput(image_id="nested/c", path=image_fixtures_dir / "nested" / "c.jpg"),
    )


def test_file_data_source_uses_recursive_flag(image_fixtures_dir: Path) -> None:
    data_source = FileDataSource(input_dir=image_fixtures_dir, recursive=False)

    images = tuple(data_source.iter_images())

    assert images == (
        ImageInput(image_id="a", path=image_fixtures_dir / "a.jpg"),
        ImageInput(image_id="b", path=image_fixtures_dir / "b.jpg"),
    )


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


def test_file_data_source_iter_images_is_repeatable(image_fixtures_dir: Path) -> None:
    data_source = FileDataSource(input_dir=image_fixtures_dir)

    first = tuple(data_source.iter_images())
    second = tuple(data_source.iter_images())

    assert first == second
