"""Schemas for reference comparison."""

from dataclasses import dataclass

from pose_deploy_gate.adapters.schema import KeypointSchema


@dataclass(frozen=True)
class SchemaResolution:
    reference_schema: KeypointSchema
    candidate_schema: KeypointSchema
    compared_keypoints: tuple[str, ...]
    reference_only_keypoints: tuple[str, ...]
    candidate_only_keypoints: tuple[str, ...]


def resolve_comparison_schema(
    reference: KeypointSchema,
    candidate: KeypointSchema,
    mode: str,
    min_common_keypoints: int,
) -> SchemaResolution:
    """Resolve exact keypoint names without alias translation.

    Strict mode requires identical names and order, regardless of schema name.
    Common mode requires at least ``min_common_keypoints`` shared names.
    Compared names follow reference order; exclusions follow their own schema's order.
    Invalid settings and incompatible schemas raise ``ValueError``.
    """
    if mode not in ("strict", "common"):
        raise ValueError(f"Unsupported schema mode: {mode!r}; expected 'strict' or 'common'")
    if (
        isinstance(min_common_keypoints, bool)
        or not isinstance(min_common_keypoints, int)
        or min_common_keypoints < 1
    ):
        raise ValueError("min_common_keypoints must be a positive integer")

    if mode == "strict" and reference.keypoint_names != candidate.keypoint_names:
        raise ValueError("Strict schema comparison requires identical keypoint names and order")

    reference_names = set(reference.keypoint_names)
    candidate_names = set(candidate.keypoint_names)
    compared = tuple(name for name in reference.keypoint_names if name in candidate_names)
    if mode == "common" and len(compared) < min_common_keypoints:
        raise ValueError(
            f"Schemas share {len(compared)} keypoints; at least {min_common_keypoints} are required"
        )

    return SchemaResolution(
        reference_schema=reference,
        candidate_schema=candidate,
        compared_keypoints=compared,
        reference_only_keypoints=tuple(
            name for name in reference.keypoint_names if name not in candidate_names
        ),
        candidate_only_keypoints=tuple(
            name for name in candidate.keypoint_names if name not in reference_names
        ),
    )
