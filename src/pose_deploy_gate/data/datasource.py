"""Data source abstractions for discovering image inputs."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pose_deploy_gate.adapters.types import ImageInput
from pose_deploy_gate.config.models import DataConfig
from pose_deploy_gate.data.file_discovery import discover_files


@dataclass(frozen=True)
class FileDataSource:
    """File-backed data source that yields normalized image inputs."""

    input_dir: Path
    file_pattern: str = "*"
    recursive: bool = False

    @classmethod
    def from_config(cls, config: DataConfig) -> "FileDataSource":
        """Build a file data source directly from validated config."""
        return cls(
            input_dir=config.input_dir,
            file_pattern=config.file_pattern,
            recursive=config.recursive,
        )

    def iter_images(self) -> Iterator[ImageInput]:
        """Yield discovered files as adapter-facing image inputs.

        This method delegates all filesystem validation and deterministic
        ordering to ``discover_files`` so the data source only handles the
        transformation from ``Path`` to ``ImageInput``.
        """
        for path in discover_files(
            input_dir=self.input_dir,
            file_pattern=self.file_pattern,
            recursive=self.recursive,
        ):
            yield ImageInput(
                image_id=self._make_image_id(path),
                path=path,
            )

    def _make_image_id(self, path: Path) -> str:
        """Create a stable image identifier from the file's relative path."""
        return path.relative_to(self.input_dir).with_suffix("").as_posix()
