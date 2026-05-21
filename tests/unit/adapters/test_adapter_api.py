from pose_deploy_gate.adapters import (
    AdapterError,
    AdapterExecutionError,
    AdapterOutput,
    DummyAdapter,
    ImageInput,
    Keypoint,
    PoseAdapter,
    PosePrediction,
    UnsupportedAdapterError,
    create_adapter,
)


def test_adapter_public_api_imports_expected_symbols():
    assert PoseAdapter is not None
    assert DummyAdapter is not None
    assert create_adapter is not None
    assert AdapterError is not None
    assert UnsupportedAdapterError is not None
    assert AdapterExecutionError is not None
    assert AdapterOutput is not None
    assert ImageInput is not None
    assert Keypoint is not None
    assert PosePrediction is not None
