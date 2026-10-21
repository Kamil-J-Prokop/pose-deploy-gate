from pathlib import Path

import pytest
from pydantic import ValidationError

from pose_deploy_gate.config import AppConfig, load_config
from pose_deploy_gate.config.models import ComparisonConfig


def test_comparison_defaults():
    assert ComparisonConfig().model_dump() == {
        "schema_mode": "common",
        "coordinate_space": "image_2d_normalized",
        "distance_metric": "normalized_euclidean_2d",
        "min_common_keypoints": 3,
        "min_pose_shared_keypoints": 3,
        "min_pose_shared_fraction": 0.5,
        "max_pose_match_distance": 0.15,
    }


def test_app_config_defaults_comparison_for_single_model():
    config = AppConfig(
        version=1,
        data={"input_dir": "."},
        models=[{"id": "model", "adapter": {"type": "dummy"}}],
    )
    assert config.comparison == ComparisonConfig()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_mode", "strict"),
        ("schema_mode", "common"),
        ("coordinate_space", "image_2d_normalized"),
        ("distance_metric", "normalized_euclidean_2d"),
        ("min_common_keypoints", 1),
        ("min_common_keypoints", 100),
        ("min_pose_shared_keypoints", 1),
        ("min_pose_shared_keypoints", 100),
        ("min_pose_shared_fraction", 0.001),
        ("min_pose_shared_fraction", 1),
        ("max_pose_match_distance", 0),
        ("max_pose_match_distance", 1),
    ],
)
def test_accepts_comparison_values(field, value):
    assert getattr(ComparisonConfig(**{field: value}), field) == value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_mode", "other"),
        ("coordinate_space", "camera_3d_metric"),
        ("coordinate_space", "world_3d_metric"),
        ("coordinate_space", "root_relative_3d"),
        ("distance_metric", "euclidean_3d"),
        ("min_common_keypoints", 0),
        ("min_common_keypoints", -1),
        ("min_common_keypoints", 1.5),
        ("min_pose_shared_keypoints", 0),
        ("min_pose_shared_keypoints", -1),
        ("min_pose_shared_keypoints", 1.5),
        ("min_pose_shared_fraction", 0),
        ("min_pose_shared_fraction", -0.1),
        ("min_pose_shared_fraction", 1.001),
        ("min_pose_shared_fraction", float("nan")),
        ("min_pose_shared_fraction", float("inf")),
        ("max_pose_match_distance", -0.001),
        ("max_pose_match_distance", 1.001),
        ("max_pose_match_distance", float("nan")),
        ("max_pose_match_distance", float("inf")),
        ("unknown", True),
    ],
)
def test_rejects_invalid_comparison_values(field, value):
    with pytest.raises(ValidationError) as exc_info:
        ComparisonConfig(**{field: value})
    assert exc_info.value.errors()[0]["loc"] == (field,)


def test_loads_comparison_overrides_from_yaml(tmp_path: Path):
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(
        f'version: 1\ndata:\n  input_dir: "{tmp_path}"\n'
        "models:\n  - id: model\n    adapter:\n      type: dummy\n"
        "comparison:\n  schema_mode: strict\n  min_common_keypoints: 1\n"
        "  min_pose_shared_keypoints: 2\n  min_pose_shared_fraction: 1\n"
        "  max_pose_match_distance: 0\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.comparison == ComparisonConfig(
        schema_mode="strict",
        min_common_keypoints=1,
        min_pose_shared_keypoints=2,
        min_pose_shared_fraction=1,
        max_pose_match_distance=0,
    )
