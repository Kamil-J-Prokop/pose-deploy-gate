from dataclasses import dataclass


@dataclass(frozen=True)
class KeypointSchema:
    name: str
    keypoint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Schema name must not be empty")
        if not self.keypoint_names:
            raise ValueError("Keypoint list must not be empty")
        if any(not name.strip() for name in self.keypoint_names):
            raise ValueError("Keypoint names must not be empty")
        if len(set(self.keypoint_names)) != len(self.keypoint_names):
            raise ValueError("Keypoint names must be unique")


DUMMY_5_SCHEMA = KeypointSchema(
    name="dummy_5",
    keypoint_names=(
        "nose",
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
    ),
)

COCO_17_SCHEMA = KeypointSchema(
    name="coco_17",
    keypoint_names=(
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ),
)
