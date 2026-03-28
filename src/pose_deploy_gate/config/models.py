from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class OutputConfig(StrictBaseModel):
    dir: Path = Path("./artifacts")


class GatesConfig(StrictBaseModel):
    enabled: bool = False


class AppConfig(StrictBaseModel):
    version: Literal[1]

    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig
    adapter: AdapterConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
