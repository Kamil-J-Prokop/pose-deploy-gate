"""Abstract adapter interface."""

from abc import ABC, abstractmethod

from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput


class PoseAdapter(ABC):
    """Common contract that every pose-model adapter must satisfy.

    Keep this surface area intentionally small for now: concrete adapters only
    need to identify themselves and return normalized predictions for one image.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a stable adapter identifier.

        Concrete adapters should implement this with a short name that can be
        used in logs, metadata, and future adapter selection code.
        """

    @abstractmethod
    def predict(self, image: ImageInput) -> AdapterOutput:
        """Run inference for a single normalized image input.

        Concrete adapters are responsible for translating their model-specific
        output into the shared :class:`AdapterOutput` structure.
        """


__all__ = ["PoseAdapter"]
