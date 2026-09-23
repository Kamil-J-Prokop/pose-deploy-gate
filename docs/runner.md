# Runner

The runner is the execution layer between configured input discovery and the
metrics, gate, and reporting stages. It executes one configured adapter over
the discovered inputs, performs optional warmup calls, captures monotonic
timings, and returns a structured `RunResult`.

## What the runner does

For a config-driven CLI invocation, PoseDeployGate:

1. loads and validates the config
2. creates the configured adapter and data source
3. discovers and materializes the input list in deterministic order
4. warms up the adapter using the first input
5. calls the adapter once for every discovered input
6. returns prediction results, failure details, and timing measurements

The CLI runs this pipeline with:

```bash
uv run python -m pose_deploy_gate --config ./path/to/config.yaml
```

After a successful run, the CLI passes the returned `RunResult` to
`MetricsEngine` and prints latency and reliability metrics. Internally, runner
timings and metric latency values are stored in nanoseconds.

## Runner results and metrics

`RunResult` is the source of raw per-attempt timings, prediction outputs, and
failure details. It describes what happened during execution without defining
deployment percentile or error-rate semantics.

`MetricsEngine` derives deployment metrics from those raw results. It excludes
failed attempts from latency samples, includes every measured attempt in the
error-rate denominator, and computes minimum, mean, P50, P95, P99, and maximum
latency. See [metrics.md](metrics.md) for the complete definitions and empty-run
behavior.

## Configuration

The runner policy is configured with the optional `runner` section:

```yaml
runner:
  warmup_iterations: 3
  continue_on_error: false
```

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `runner.warmup_iterations` | non-negative integer | `3` | Number of adapter calls made before measured predictions. Use `0` to disable warmup. |
| `runner.continue_on_error` | boolean | `false` | Whether to record an adapter prediction failure and continue with later inputs. |

The minimal config can omit this section and use both defaults. See
[config.minimal.yaml](examples/config.minimal.yaml) and the explicit runner
example in [config.runner.yaml](examples/config.runner.yaml).

## Warmup

Warmup calls the adapter repeatedly with the first discovered input before any
measured prediction is made. The returned adapter outputs are discarded.

Warmup is useful for runtimes that perform one-time initialization during their
first calls, such as allocating buffers, loading kernels, or populating caches.
It reduces the influence of those costs on the per-prediction measurements.
The runner does not guarantee that every adapter has reached a stable state;
the appropriate iteration count depends on the adapter and runtime.

Warmup behavior has these properties:

- `warmup_iterations: 0` disables warmup.
- Every warmup iteration uses the first input.
- `warmup.total_time_ns` records the duration of the complete warmup block.
- Warmup time is excluded from measured and average inference time.
- Warmup time is included in `run.total_time_ns`.
- An `AdapterExecutionError` during warmup always stops the run and is wrapped
  in `RunnerExecutionError`, regardless of `continue_on_error`.

## Timing capture

The runner uses Python's monotonic `time.perf_counter_ns()` clock. These values
are durations rather than wall-clock timestamps and are suitable for measuring
elapsed time even if the system clock changes during a run.

Each measured prediction is timed around the complete `adapter.predict(image)`
call. Consequently, a prediction measurement includes all work performed by
the adapter inside that call. Depending on the adapter, that may include image
loading, preprocessing, model inference, synchronization, postprocessing, and
adapter-level bookkeeping. It is not necessarily accelerator kernel time.

### Timing fields

The names below use `run` for the returned `RunResult`, `warmup` for its
`WarmupResult`, and `prediction` for one `PredictionResult`.

| Field | Type | Meaning |
| --- | --- | --- |
| `warmup.total_time_ns` | integer | Total elapsed time for all warmup calls. It is `0` when warmup does not run. |
| `prediction.timing.elapsed_ns` | integer | Elapsed time for one adapter prediction attempt. Failed attempts recorded under the continue policy also have a timing. |
| `run.total_time_ns` | integer | Total measured runner execution window after input materialization, including warmup and prediction orchestration through the last attempt. |
| `run.measured_inference_time_ns` | integer | Sum of every prediction attempt's `elapsed_ns`, including recorded failed attempts and excluding warmup. |
| `run.average_inference_time_ns` | float | Mean `elapsed_ns` across successful predictions only. It is `0.0` when there are no successful predictions. |

The CLI converts these nanosecond values to milliseconds with `ns_to_ms()` and
prints three digits after the decimal point. Timing values naturally vary
between runs, even when adapter outputs and input ordering are deterministic.

## What timing excludes

Input discovery and materialization occur before the runner starts its total
timer. The runner timing fields therefore exclude:

- config loading and validation
- runner, adapter, and data source construction
- filesystem input discovery and creation of `ImageInput` values
- CLI output formatting and printing
- any later metrics computation, gate evaluation, report writing, or artifact persistence

`run.total_time_ns` includes the warmup block and orchestration between adapter
calls inside the measured execution window. It is therefore broader than
`run.measured_inference_time_ns`, which is the sum of prediction attempts only.
Warmup has its own total and is not included in either inference aggregate.

## Failure policy

The failure policy applies to `AdapterExecutionError` raised during measured
predictions:

- With `continue_on_error: false`, the runner stops at the first failed
  prediction and raises `RunnerExecutionError`. No `RunResult` is returned.
- With `continue_on_error: true`, the runner stores a `PredictionResult` with
  `output=None`, the error message, and the failed attempt's elapsed time, then
  continues with the next input.

Recorded failures count toward `run.failed_predictions` and
`run.measured_inference_time_ns`. They do not count toward
`run.average_inference_time_ns`, whose denominator contains successful
predictions only.

Warmup failures are always fatal. Configuration, adapter construction, and
data source failures also remain errors from their respective layers. The CLI
prints these failures with an `ERROR:` prefix and returns exit code `2`.

## Intentionally out of scope

The runner deliberately does not provide:

- latency percentiles or error rates directly; `MetricsEngine` derives them
  from `RunResult`
- pose-output validation or accuracy metrics
- deployment gate evaluation
- JSON or Markdown report generation
- artifact persistence
- batching, parallel execution, or asynchronous adapters
- retries, timeouts, or recovery from unexpected exception types
- CPU, GPU, memory, power, or throughput measurements
- separate preprocessing, model-kernel, and postprocessing timings

Keeping the runner focused provides one stable source of prediction results and
raw elapsed-time measurements for the metrics, validation, gate, and reporting
layers.
