"""Issues for adapter output validation."""

from dataclasses import dataclass
from enum import StrEnum


class OutputValidationCode(StrEnum):
    """Stable identifiers for output validation failures."""

    COORDINATE_OUT_OF_RANGE = "coordinate_out_of_range"
    INVALID_COORDINATE_PAIR = "invalid_coordinate_pair"
    NON_FINITE_COORDINATE = "non_finite_coordinate"
    VISIBLE_MISSING_KEYPOINT = "visible_missing_keypoint"


@dataclass(frozen=True)
class OutputValidationIssue:
    """One validation failure at a specific location in an adapter output."""

    path: str
    code: OutputValidationCode
    message: str
