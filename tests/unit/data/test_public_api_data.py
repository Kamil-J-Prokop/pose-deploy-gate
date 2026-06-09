from pose_deploy_gate.data import (
    DataSourceError,
    FileDataSource,
    InputDirectoryError,
    NoInputFilesError,
    create_data_source,
    discover_files,
)


def test_data_public_api_imports_expected_symbols() -> None:
    assert DataSourceError is not None
    assert FileDataSource is not None
    assert InputDirectoryError is not None
    assert NoInputFilesError is not None
    assert create_data_source is not None
    assert discover_files is not None
