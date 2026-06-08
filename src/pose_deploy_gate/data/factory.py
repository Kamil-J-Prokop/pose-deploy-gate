"""Data source creation from config."""

from pose_deploy_gate.config.models import DataConfig
from pose_deploy_gate.data.datasource import FileDataSource


def create_data_source(config: DataConfig) -> FileDataSource:
    """Create a file-backed data source from validated config."""
    return FileDataSource.from_config(config)
