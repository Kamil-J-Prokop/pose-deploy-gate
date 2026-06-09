from pathlib import Path

import pytest


@pytest.fixture
def image_fixtures_dir() -> Path:
    """Return the shared image fixture directory used by data tests."""
    return Path(__file__).parent / "fixtures" / "data" / "images"
