from pose_deploy_gate.adapters.exceptions import (
    AdapterError,
    AdapterExecutionError,
    UnsupportedAdapterError,
)


def test_unsupported_adapter_error_is_adapter_error():
    error = UnsupportedAdapterError("dummy")

    assert isinstance(error, AdapterError)


def test_adapter_execution_error_is_adapter_error():
    error = AdapterExecutionError("prediction failed")

    assert isinstance(error, AdapterError)
