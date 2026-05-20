from typing import cast

import pytest

from pose_deploy_gate.adapters.dummy import DummyAdapter
from pose_deploy_gate.adapters.exceptions import AdapterError, UnsupportedAdapterError
from pose_deploy_gate.adapters.factory import create_adapter
from pose_deploy_gate.config.models import AdapterConfig


class FakeAdapterConfig:
    def __init__(self, adapter_type: str, params: dict[str, object] | None = None) -> None:
        self.type = adapter_type
        self.params = {} if params is None else params


def test_create_adapter_returns_dummy_adapter():
    config = AdapterConfig(type="dummy")

    adapter = create_adapter(config)

    assert isinstance(adapter, DummyAdapter)


def test_create_adapter_passes_dummy_params():
    config = AdapterConfig(
        type="dummy",
        params={"keypoint_confidence": 0.9, "pose_confidence": 0.8},
    )

    adapter = create_adapter(config)

    assert isinstance(adapter, DummyAdapter)
    assert adapter.keypoint_confidence == 0.9
    assert adapter.pose_confidence == 0.8


def test_create_adapter_raises_for_unsupported_adapter_type():
    config = cast(AdapterConfig, FakeAdapterConfig("not-supported"))

    with pytest.raises(UnsupportedAdapterError, match="Unsupported adapter type: not-supported"):
        create_adapter(config)


def test_create_adapter_wraps_invalid_params_as_adapter_error():
    config = AdapterConfig(type="dummy", params={"unexpected_param": 123})

    with pytest.raises(AdapterError, match="Invalid params for adapter 'dummy'") as exc_info:
        create_adapter(config)

    assert isinstance(exc_info.value.__cause__, TypeError)
