from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictBaseModel):
    name: str = "default-run"
    seed: int | None = None
    device: Literal["cpu", "cuda", "auto"] = "cpu"


class DataConfig(StrictBaseModel):
    input_dir: Path
    file_pattern: str = "*"
    recursive: bool = False


class AdapterConfig(StrictBaseModel):
    type: Literal["dummy"]
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(StrictBaseModel):
    id: str
    adapter: AdapterConfig

    @field_validator("id")
    @classmethod
    def validate_id(cls, model_id: str) -> str:
        if not model_id.strip():
            raise ValueError("model ID must not be blank")
        return model_id


class OutputConfig(StrictBaseModel):
    dir: Path = Path("./artifacts")


class GatesConfig(StrictBaseModel):
    enabled: bool = False


class RunnerConfig(StrictBaseModel):
    warmup_iterations: int = Field(default=3, ge=0)
    continue_on_error: bool = False


class ComparisonConfig(StrictBaseModel):
    schema_mode: Literal["strict", "common"] = "common"
    coordinate_space: Literal["image_2d_normalized"] = "image_2d_normalized"
    distance_metric: Literal["normalized_euclidean_2d"] = "normalized_euclidean_2d"
    min_common_keypoints: int = Field(default=3, ge=1)
    min_pose_shared_keypoints: int = Field(default=3, ge=1)
    min_pose_shared_fraction: float = Field(default=0.5, gt=0, le=1)
    max_pose_match_distance: float = Field(default=0.15, ge=0, le=1)


class AppConfig(StrictBaseModel):
    version: Literal[1]

    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig
    models: tuple[ModelConfig, ...] = Field(min_length=1)
    output: OutputConfig = Field(default_factory=OutputConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)

    @field_validator("models")
    @classmethod
    def validate_model_ids(cls, models: tuple[ModelConfig, ...]) -> tuple[ModelConfig, ...]:
        ids = [model.id for model in models]
        if len(ids) != len(set(ids)):
            raise ValueError("model IDs must be unique")
        return models
