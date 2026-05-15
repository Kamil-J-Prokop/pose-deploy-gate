"""Data structures shared by all adapters."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageInput:
    """Normalized image reference passed into adapters."""

    image_id: str
    path: Path


@dataclass(frozen=True)
class Keypoint:
    """Single predicted keypoint in image coordinates."""

    name: str
    x: float | None
    y: float | None
    confidence: float | None = None
    visible: bool | None = None


@dataclass(frozen=True)
class PosePrediction:
    """Normalized pose prediction emitted by an adapter."""

    keypoints: tuple[Keypoint, ...]
    confidence: float
    person_id: str | None = None


@dataclass(frozen=True)
class AdapterOutput:
    """Top-level normalized adapter output for one image."""

    poses: tuple[PosePrediction, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = ["AdapterOutput", "ImageInput", "Keypoint", "PosePrediction"]
