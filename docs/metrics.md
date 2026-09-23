# Metrics

`MetricsEngine` derives deployment-oriented latency and reliability metrics
from the raw prediction results captured by the runner. It performs aggregation
only: it does not run inference, validate pose outputs, or decide whether a run
passes or fails.

## Public API

The metrics types are available from `pose_deploy_gate.metrics`:

```python
from pose_deploy_gate.metrics import MetricsEngine, MetricsResult
from pose_deploy_gate.runner import RunResult


def compute_metrics(run_result: RunResult) -> MetricsResult:
    return MetricsEngine().compute(run_result)
```

`MetricsEngine.compute()` accepts one immutable `RunResult` and returns a
`MetricsResult` containing:

- `latency`: a `LatencyMetrics` value
- `errors`: an `ErrorRateMetrics` value

Computing metrics does not mutate the supplied `RunResult`. Repeated calls with
the same result produce equal metrics, and prediction ordering does not affect
the aggregates.

## Latency samples

One latency sample is the `prediction.timing.elapsed_ns` value from one
successful measured `adapter.predict()` attempt. A prediction is successful
when `PredictionResult.error is None`.

Latency metrics exclude:

- adapter warmup calls
- failed prediction attempts
- filesystem discovery and input materialization
- config parsing and validation
- runner, adapter, and data source construction

The measurement still includes every operation performed inside the adapter's
`predict()` call. Depending on the adapter, that can include input loading,
preprocessing, inference, device synchronization, postprocessing, and
adapter-level bookkeeping. It is not necessarily accelerator kernel time.

All stored latency values use nanoseconds. `LatencyMetrics` reports:

| Field | Meaning |
| --- | --- |
| `sample_count` | Number of successful measured prediction attempts. |
| `min_ns` | Minimum successful prediction latency. |
| `mean_ns` | Arithmetic mean of successful prediction latencies. |
| `p50_ns` | 50th-percentile successful prediction latency. |
| `p95_ns` | 95th-percentile successful prediction latency. |
| `p99_ns` | 99th-percentile successful prediction latency. |
| `max_ns` | Maximum successful prediction latency. |

### Percentile method

Percentiles use NumPy's `percentile()` function with
`method="inverted_cdf"`. This is a discontinuous empirical-distribution method:
it selects an observed sample rather than interpolating between adjacent
samples. For a percentile fraction `q` and `n` sorted samples, it selects the
observation at the one-based rank `ceil(q * n)`.

This choice keeps percentile results tied to durations that were actually
observed. With small sample counts, high percentiles commonly equal the
maximum. For example, P95 and P99 of six samples are both the sixth sample.

### No successful predictions

When no prediction succeeds, `sample_count` is `0` and `min_ns`, `mean_ns`,
`p50_ns`, `p95_ns`, `p99_ns`, and `max_ns` are all `None`. CLI presentation
renders those undefined values as `n/a`, not `0.000 ms`.

## Error-rate metrics

The error-rate denominator is every measured prediction attempt in
`run_result.predictions`, including successful and failed attempts. Warmup calls
are not measured prediction attempts and are not part of the denominator.

A measured attempt is a failure exactly when
`PredictionResult.error is not None`.

The derived values are:

```text
total_attempts = len(run_result.predictions)
failed_predictions = number of predictions whose error is not None
successful_predictions = total_attempts - failed_predictions
error_rate = failed_predictions / total_attempts
success_rate = successful_predictions / total_attempts
```

Rates are ratios in the inclusive range `0.0` through `1.0`, not percentages.
Presentation layers may multiply them by 100; for example, the CLI renders an
error-rate ratio of `0.02` as `2.00%`.

### No prediction attempts

When a `RunResult` contains no prediction attempts, `total_attempts`,
`failed_predictions`, and `successful_predictions` are all `0`. Both
`error_rate` and `success_rate` are `None` because no denominator exists.

## No pass/fail interpretation

`MetricsEngine` reports observations and rates. It does not apply thresholds,
compare models, or determine whether deployment criteria pass. Those decisions
belong to the later gate-evaluation layer.
