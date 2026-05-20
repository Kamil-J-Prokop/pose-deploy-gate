"""Adapter creation from config"""

from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.adapters.dummy import DummyAdapter
from pose_deploy_gate.adapters.exceptions import AdapterError, UnsupportedAdapterError
from pose_deploy_gate.config.models import AdapterConfig

ADAPTER_REGISTRY: dict[str, type[PoseAdapter]] = {
    "dummy": DummyAdapter,
}


def create_adapter(config: AdapterConfig) -> PoseAdapter:
    adapter_cls = ADAPTER_REGISTRY.get(config.type)
    if adapter_cls is None:
        raise UnsupportedAdapterError(f"Unsupported adapter type: {config.type}")

    try:
        return adapter_cls(**config.params)
    except TypeError as exc:
        raise AdapterError(f"Invalid params for adapter '{config.type}': {config.params}") from exc
