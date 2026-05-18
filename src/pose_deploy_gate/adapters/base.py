"""Abstract adapter interface."""

from abc import ABC, abstractmethod

from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput


class PoseAdapter(ABC):
    """Common contract that every pose-model adapter must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable adapter identifier."""

    @abstractmethod
    def predict(self, image: ImageInput) -> AdapterOutput:
        """Run inference for a single normalized image input."""


__all__ = ["PoseAdapter"]
