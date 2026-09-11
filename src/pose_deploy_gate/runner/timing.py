"""Timer abstraction for runner timing capture."""

import time


class Timer:
    """Monotonic nanosecond timer used by runner timing capture."""

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    def elapsed_ns(self, start_ns: int) -> int:
        return self.now_ns() - start_ns


def ns_to_ms(value_ns: int) -> float:
    return value_ns / 1_000_000
