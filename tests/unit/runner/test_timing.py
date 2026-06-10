from pose_deploy_gate.runner.timing import Timer, ns_to_ms


def test_timer_now_ns_returns_int() -> None:
    timer = Timer()

    assert isinstance(timer.now_ns(), int)


def test_timer_now_ns_is_monotonic() -> None:
    timer = Timer()

    start = timer.now_ns()
    end = timer.now_ns()

    assert end >= start


def test_timer_elapsed_ns_returns_non_negative_duration() -> None:
    timer = Timer()

    start = timer.now_ns()
    elapsed = timer.elapsed_ns(start)

    assert isinstance(elapsed, int)
    assert elapsed >= 0


def test_ns_to_ms_converts_nanoseconds_to_milliseconds() -> None:
    assert ns_to_ms(1_500_000) == 1.5
