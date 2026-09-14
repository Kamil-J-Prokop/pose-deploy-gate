# PoseDeployGate v1.0.0 Release Contract

This contract defines the release finish line and prevents feature bloat and scope drift.. Checked means already fulfilled; unchecked means required before the `v1.0.0` tag.

## 1. What exact user problem does v1.0 solve?

PoseDeployGate v1.0 helps engineers determine whether a pose-model runtime meets defined deployment latency, stability, and output-validity requirements on a target Python-capable Linux machine. It replaces ad hoc benchmarking with one repeatable, configuration-driven command that produces a clear pass/fail decision and reviewable evidence for humans and CI systems.

- [ ] A technically competent engineer can run one repeatable, config-driven evaluation of a pose-model adapter over local files and receive a deployment decision without reading the source or asking the maintainer questions.
- [ ] The decision is based on deterministic input ordering, measured inference latency, inference failure rate, minimal pose-output schema validity, and configured pass/fail thresholds.
- [ ] The result is useful in a terminal and in CI: it provides human-readable output, machine-readable evidence, and stable exit codes.
- [x] PDG evaluates deployment behaviour through a Python adapter boundary; it does not deploy, train, convert, or rank models by accuracy.

## 2. What exact environment does it support?

- [x] The core package requires CPython 3.12 or newer and uses the documented `uv` installation workflow.
- [x] The verified v1.0 reference environment is 64-bit Ubuntu 24.04 with CPython 3.12, and this exact environment must pass CI.
- [x] Python runs on the benchmark machine, inputs and configuration are local files, and native runtimes may be invoked through Python bindings.
- [x] `DummyAdapter` is the dependency-light, deterministic adapter used by tests and documentation.
- [ ] One real runtime boundary is supported: `ONNXRuntimeAdapter` using ONNX Runtime CPU. Its dependency is an optional extra; no CUDA provider is promised.
- [ ] A documented, locally reproducible ONNX example is included; it requires no credentials, remote service, or device orchestration during the benchmark run.
- [x] Systems without Python, bare embedded targets, Windows, GPUs, and environments not covered by the release CI are outside the v1.0 support promise.

## 3. What exact end-to-end workflow must work?

- [x] Load a documented YAML configuration, reject unknown or invalid values before execution, and report missing configuration or input paths clearly.
- [x] Discover matching local files and deliver them to the selected adapter in stable relative-POSIX-path order.
- [x] Construct the selected adapter through the public factory/interface, run configurable warm-up calls, and exclude warm-up from measured inference results.
- [x] Time every measured `predict()` attempt with a monotonic clock and preserve successful and failed attempt counts under the configured failure policy.
- [ ] Validate each adapter result against a documented, mechanically testable minimal pose-output schema covering container structure, keypoint dimensions, required fields, numeric types, finite values, and the treatment of zero detected poses. Distinguish inference failures from successfully returned but invalid outputs.
- [ ] Compute p50, p95, and p99 over measured `predict()` calls that return without raising, including calls that return invalid outputs, using the documented nearest-rank method and millisecond units. If no measured call returns, latency percentiles are `null` and all latency gates fail.
- [ ] Evaluate gates using documented inclusive comparisons: latency percentile `<=` configured maximum, inference failure rate `= inference failures / measured attempts`, and invalid-output rate `= invalid outputs / measured attempts`. Invalid outputs and inference failures are separate categories.
- [ ] From a clean checkout, `uv sync --dev --extra onnx` followed by `uv run python -m pose_deploy_gate --config docs/examples/config.onnx.yaml --output-dir artifacts/v1` completes without undocumented setup.
- [ ] That command writes `artifacts/v1/report.json` and `artifacts/v1/summary.md` with stable schemas, deterministic field and input ordering, and documented volatile fields such as latency measurements and environment metadata.
- [ ] Both reports contain `"schema_version": "1.0"`, measured count, p50/p95/p99 latency, failure and invalid-output details, gate results, normalized configuration, its SHA-256 digest, PDG/Python/OS/runtime metadata, and the documented timing boundary. The digest is calculated from the documented canonical JSON representation, with defaults expanded and volatile output-path fields excluded.
- [ ] If adapter construction succeeds, its documented `close()` lifecycle method is called exactly once after successful completion, gate failure, or any handled failure. Cleanup errors must not hide an earlier primary error.

## 4. What evidence proves it works?

- [x] Unit tests cover strict configuration validation, deterministic discovery, adapter contracts, warm-up exclusion, per-call timing, and runner failure policy.
- [ ] The complete test suite passes on the release-candidate commit in the verified reference environment.
- [ ] Unit tests deterministically verify p50/p95/p99 calculation, rate denominators, gate boundaries, stable exit codes, report schemas, deterministic ordering, and all report fields documented as non-volatile.
- [ ] One integration test executes the exact command path from config through ONNX Runtime CPU to both report files and checks pass and fail outcomes.
- [x] CI on the reference Linux environment runs Ruff formatting/linting through pre-commit and the pytest suite.
- [ ] CI enforces at least 85% line and 70% branch coverage for core evaluation modules, builds the wheel, installs it in a clean environment, and runs the integration smoke test.
- [x] Package version metadata and the Apache-2.0 license are present.
- [ ] README/docs provide a five-minute quickstart, config and report references, public adapter API, architecture diagram, timing semantics, stable exit codes, supported environment, limitations, and example reports.
- [ ] A release-candidate audit proves a clean clone can install, run the exact example, inspect the reports and unsupported targets, and pass the full test suite; no open defect breaks that path or corrupts results.

## 5. What tempting work is explicitly excluded?

- [x] Direct execution without Python; a native C/C++ agent; bare embedded support; and remote-device orchestration.
- [x] Any second real inference framework, automatic plugin discovery, arbitrary import-path plugins, or multi-model comparison in one run.
- [x] Model training, dataset preparation, accuracy evaluation, quantization, model conversion, or canonical keypoint-taxonomy design beyond the minimal output validator.
- [x] CUDA/GPU support, kernel profiling, memory or energy measurement, throughput/load testing, batching, parallelism, streaming, and asynchronous execution.
- [x] Cloud deployment, hosted dashboards, remote inference services, JUnit output, or report formats other than console, JSON, and Markdown.
- [x] Bugs or documentation required by an unchecked item may enter v1.0; new capabilities may not. An essential new requirement must replace work of similar effort.
- [ ] `BACKLOG.md` exists and holds all other feedback and attractive ideas for v1.1 or v2.0.

## End-to-end demonstration acceptance criteria

- [ ] Starting from a clean clone on the reference Linux environment, a technically competent engineer can follow the README quickstart to install PDG without consulting additional documentation, editing project files, or asking the maintainer questions.
- [ ] The engineer can run `uv run python -m pose_deploy_gate --config docs/examples/config.onnx.yaml --output-dir artifacts/v1` without editing code, supplying credentials, downloading an undocumented asset, or asking the maintainer questions.
- [ ] The run exercises local deterministic input discovery, ONNX Runtime CPU adapter creation, warm-up, measured inference, output validation, metrics, gates, and report writing in one process.
- [ ] The console identifies the timing boundary and prints measured count, p50/p95/p99 latency, failure and invalid-output rates, every gate result, and the overall PASS/FAIL decision.
- [ ] The run exits `0` and produces `artifacts/v1/report.json` and `artifacts/v1/summary.md`; both reports agree with the console and contain the required configuration digest and environment metadata.
- [ ] Repeating the run with unchanged configuration and inputs preserves input ordering, counts, decisions, normalized configuration digest, schema version, and report structure. Latency values and explicitly documented volatile environment fields may vary.
- [ ] Return `0` after a completed passing evaluation, `1` after a completed evaluation that fails any gate, and `2` when invalid configuration or a fatal setup/execution error prevents completion. Per-input inference exceptions are recorded as inference failures and do not independently cause exit code `2`.
- [ ] A documented gate-failing example returns `1`, identifies every failed threshold in the console and both reports, and still emits complete evaluation evidence.
- [ ] A documented invalid-configuration example returns `2`, provides an actionable error, performs no inference, and is not required to emit evaluation reports.
- [ ] An engineer can explain what PDG measured, where timing occurred, why the gate passed or failed, and which targets v1.0 does not support.
- [ ] The same demonstration can be performed live or recorded in three minutes after installation, with no manual step omitted from the documentation.
