"""NumPy-backed statistics module for percentile computation."""

from collections.abc import Sequence

import numpy as np


def compute_percentiles(
    values: Sequence[float | int],
) -> tuple[float | None, float | None, float | None]:
    if len(values) == 0:
        return None, None, None

    p50, p95, p99 = np.percentile(
        values,
        [50, 95, 99],
        method="inverted_cdf",
    )

    return float(p50), float(p95), float(p99)
