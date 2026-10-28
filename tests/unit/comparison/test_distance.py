from math import sqrt

import pytest

from pose_deploy_gate.adapters.types import Keypoint
from pose_deploy_gate.comparison import normalized_image_2d_distance


def _point(x, y):
    return Keypoint(name="nose", x=x, y=y)


@pytest.mark.parametrize("x,y", [(0, 0), (0.3, 0.7), (1, 1)])
def test_identical_points_have_zero_distance(x, y):
    point = _point(x, y)
    assert normalized_image_2d_distance(point, point) == 0


@pytest.mark.parametrize("reference,candidate", [((0, 0), (1, 1)), ((0, 1), (1, 0))])
def test_opposite_corners_have_unit_distance(reference, candidate):
    assert normalized_image_2d_distance(_point(*reference), _point(*candidate)) == 1


def test_known_diagonal_distance():
    assert normalized_image_2d_distance(_point(0.1, 0.2), _point(0.4, 0.6)) == pytest.approx(
        0.5 / sqrt(2)
    )


@pytest.mark.parametrize(
    "reference,candidate", [((0, 0), (0.5, 0.5)), ((0, 1), (1, 1)), ((0.2, 0.7), (0.9, 0.3))]
)
def test_distance_is_symmetric_and_bounded(reference, candidate):
    reference, candidate = _point(*reference), _point(*candidate)
    distance = normalized_image_2d_distance(reference, candidate)
    assert distance == normalized_image_2d_distance(candidate, reference)
    assert 0 <= distance <= 1


@pytest.mark.parametrize("coordinates", [(None, None), (None, 0.5), (0.5, None)])
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_missing_coordinates_are_rejected(coordinates, side):
    missing, valid = _point(*coordinates), _point(0, 0)
    reference, candidate = (missing, valid) if side == "reference" else (valid, missing)
    with pytest.raises(ValueError, match=f"{side} keypoint requires both"):
        normalized_image_2d_distance(reference, candidate)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_invalid_coordinates_are_rejected(value, axis, side):
    invalid = _point(value, 0.5) if axis == "x" else _point(0.5, value)
    valid = _point(0, 0)
    reference, candidate = (invalid, valid) if side == "reference" else (valid, invalid)
    with pytest.raises(ValueError, match=f"{side} keypoint coordinates must be finite"):
        normalized_image_2d_distance(reference, candidate)
