from pose_deploy_gate.metrics.statistics import compute_percentiles


def test_compute_percentiles_returns_none_for_empty_list() -> None:
    p50, p95, p99 = compute_percentiles([])

    assert p50 is None
    assert p95 is None
    assert p99 is None


def test_compute_percentiles_returns_correct_values() -> None:
    values = list(range(1, 101))
    assert compute_percentiles(values) == (50.0, 95.0, 99.0)


def test_compute_percentiles_returns_correct_values_with_duplicates() -> None:
    values = [1, 2, 2, 3, 4, 5]
    p50, p95, p99 = compute_percentiles(values)

    assert p50 == 2.0
    assert p95 == 5.0
    assert p99 == 5.0


def test_compute_percentiles_returns_correct_values_with_the_same_value() -> None:
    values = [100, 100, 100, 100, 100]
    p50, p95, p99 = compute_percentiles(values)

    assert p50 == 100.0
    assert p95 == 100.0
    assert p99 == 100.0


def test_compute_percentiles_returns_correct_values_with_large_numbers() -> None:
    values = [1000, 2000, 3000, 4000, 5000]
    p50, p95, p99 = compute_percentiles(values)

    assert p50 == 3000.0
    assert p95 == 5000.0
    assert p99 == 5000.0


def test_compute_percentiles_returns_the_same_value_for_unsorted_input() -> None:
    unsorted_values = [5, 1, 3, 2, 4]
    us_p50, us_p95, us_p99 = compute_percentiles(unsorted_values)

    sorted_values = [1, 2, 3, 4, 5]
    s_p50, s_p95, s_p99 = compute_percentiles(sorted_values)

    assert us_p50 == s_p50
    assert us_p95 == s_p95
    assert us_p99 == s_p99
