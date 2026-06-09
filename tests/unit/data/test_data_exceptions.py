from pose_deploy_gate.data.exceptions import (
    DataSourceError,
    InputDirectoryError,
    NoInputFilesError,
)


def test_input_directory_error_is_data_source_error() -> None:
    assert issubclass(InputDirectoryError, DataSourceError)


def test_no_input_files_error_is_data_source_error() -> None:
    assert issubclass(NoInputFilesError, DataSourceError)
