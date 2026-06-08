"""Helpers for discovering input files from a directory."""

from pathlib import Path

from pose_deploy_gate.data.exceptions import InputDirectoryError, NoInputFilesError


def _relative_posix_sort_key(path: Path, input_dir: Path) -> str:
    """Return a normalized relative path string for deterministic sorting."""
    return path.relative_to(input_dir).as_posix()


def discover_files(
    input_dir: Path,
    file_pattern: str = "*",
    recursive: bool = False,
) -> tuple[Path, ...]:
    """Discover matching files in a directory with deterministic ordering.

    Sorting by the relative POSIX path keeps results stable across platforms.
    Native path ordering can differ between operating systems because of path
    separators, so we normalize before sorting.
    """
    if not input_dir.exists():
        raise InputDirectoryError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise InputDirectoryError(f"Input path is not a directory: {input_dir}")

    iterator = input_dir.rglob(file_pattern) if recursive else input_dir.glob(file_pattern)

    files = tuple(
        sorted(
            (path for path in iterator if path.is_file()),
            key=lambda path: _relative_posix_sort_key(path, input_dir),
        )
    )

    if not files:
        raise NoInputFilesError(
            f"No input files found in {input_dir} matching pattern {file_pattern!r}"
        )

    return files
