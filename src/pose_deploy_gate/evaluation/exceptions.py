"""Errors in the correspondence between model evaluation runs."""


class EvaluationConsistencyError(ValueError):
    """Model predictions cannot be aligned unambiguously by image ID."""
