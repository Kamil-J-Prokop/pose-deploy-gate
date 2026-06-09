from pathlib import Path

from pose_deploy_gate.config.models import DataConfig
from pose_deploy_gate.data.datasource import FileDataSource
from pose_deploy_gate.data.factory import create_data_source


def test_create_data_source_returns_file_data_source(tmp_path: Path) -> None:
    config = DataConfig(input_dir=tmp_path)

    data_source = create_data_source(config)

    assert isinstance(data_source, FileDataSource)


def test_create_data_source_uses_config_values(tmp_path: Path) -> None:
    config = DataConfig(
        input_dir=tmp_path,
        file_pattern="*.png",
        recursive=True,
    )

    data_source = create_data_source(config)

    assert data_source == FileDataSource(
        input_dir=tmp_path,
        file_pattern="*.png",
        recursive=True,
    )
