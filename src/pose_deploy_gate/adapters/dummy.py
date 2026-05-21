"""Deterministic fake adapter."""

from dataclasses import dataclass

from pose_deploy_gate.adapters.base import PoseAdapter
from pose_deploy_gate.adapters.types import AdapterOutput, ImageInput, Keypoint, PosePrediction


@dataclass(frozen=True)
class DummyAdapter(PoseAdapter):
    """Predictable adapter used for tests and local wiring."""

    keypoint_confidence: float = 1.0
    pose_confidence: float = 1.0

    @property
    def name(self) -> str:
        """Return the stable adapter identifier."""
        return "dummy"

    def predict(self, image: ImageInput) -> AdapterOutput:
        """Return a fixed pose prediction with stable coordinates."""
        pose = PosePrediction(
            keypoints=(
                Keypoint(name="nose", x=0.50, y=0.10, confidence=self.keypoint_confidence),
                Keypoint(name="left_shoulder", x=0.40, y=0.30, confidence=self.keypoint_confidence),
                Keypoint(
                    name="right_shoulder", x=0.60, y=0.30, confidence=self.keypoint_confidence
                ),
                Keypoint(name="left_hip", x=0.45, y=0.55, confidence=self.keypoint_confidence),
                Keypoint(name="right_hip", x=0.55, y=0.55, confidence=self.keypoint_confidence),
            ),
            confidence=self.pose_confidence,
            person_id="dummy-person-0",
        )

        return AdapterOutput(
            poses=(pose,),
            metadata={
                "image_id": image.image_id,
                "adapter_name": self.name,
            },
        )
