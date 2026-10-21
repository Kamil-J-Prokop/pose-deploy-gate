"""Reference comparison package."""

from .distance import normalized_image_2d_distance
from .schema import SchemaResolution, resolve_comparison_schema

__all__ = ["SchemaResolution", "normalized_image_2d_distance", "resolve_comparison_schema"]
