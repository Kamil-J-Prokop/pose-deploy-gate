"""Distance calculations for reference comparison."""

from math import hypot, isfinite, sqrt

from pose_deploy_gate.adapters.types import Keypoint


def normalized_image_2d_distance(reference: Keypoint, candidate: Keypoint) -> float:
    """Return Euclidean image distance divided by the unit square's diagonal.

    Both points must have finite coordinates in [0, 1]; invalid or missing
    coordinates raise ValueError. The comparator decides whether to skip missing
    points. Names, confidence, and visibility do not affect this calculation.
    """
    for label, point in (("reference", reference), ("candidate", candidate)):
        if point.x is None or point.y is None:
            raise ValueError(f"{label} keypoint requires both x and y coordinates")
        if any(not isfinite(value) or not 0 <= value <= 1 for value in (point.x, point.y)):
            raise ValueError(f"{label} keypoint coordinates must be finite and in [0, 1]")

    return hypot(reference.x - candidate.x, reference.y - candidate.y) / sqrt(2)
