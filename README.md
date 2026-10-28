If you have ever asked yourself:
>Can I swap my trusted **pose estimation** model or runtime for this cheaper/faster one without materially changing my application's outputs?

PoseDeployGate is being built to answer exactly that question.

# PoseDeployGate

Deployment-first evaluation and selection framework for human pose estimation models under real-world constraints.

## Purpose

PoseDeployGate is a practical framework for comparing human pose estimation models beyond raw benchmark accuracy.

The goal is to support model selection under deployment constraints such as:

- latency
- runtime stability
- input robustness
- deployment environment compatibility
- reproducibility of evaluation

The project focuses on closing the gap between research-grade metrics and production deployment decisions.

## Example output

```
PoseDeployGate config validation successful.
Config path: /path/to/config.comparison.yaml
Run name: default-run
Input directory: /path/to/inputs
Output directory: artifacts
Gates enabled: False
PoseDeployGate evaluation completed.

Reference: production
  Images: 3
  P95 latency: 0.041 ms
  Failure rate: 0.00%
  Successful predictions: 3
  Failed predictions: 0
    Adapter execution failures: 0
    Output validation failures: 0

Candidate: candidate
  Images: 3
  P95 latency: 0.025 ms
  Failure rate: 0.00%
  Successful predictions: 3
  Failed predictions: 0
    Adapter execution failures: 0
    Output validation failures: 0

  Schema comparison:
    Reference: dummy_5
    Candidate: dummy_5
    Compared keypoints: 5/5 reference, 5/5 candidate
    Reference-only keypoints excluded: 0
    Candidate-only keypoints excluded: 0

  Quality:
    Comparison coverage: 100.00%
    Matched poses: 3
    Missed reference poses: 0 (0.00%)
    Extra candidate poses: 0
    Mean keypoint deviation: 0.00%
    P95 keypoint deviation: 0.00%
    Missing keypoint rate: 0.00%

```

Timings above are illustrative. The first configured model is the reference;
each remaining candidate gets its own section in config order. See the
[multi-model config example](docs/examples/config.comparison.yaml) and
[evaluation semantics](docs/evaluation.md).

For detailed runtime statistics on every model, add `-v` (`--verbose` and
`-verbose` are also accepted). Run this two-model smoke test from the repository
root:

```bash
uv run python -m pose_deploy_gate --config docs/examples/config.comparison-smoke.yaml -v
```

This uses two separate dummy adapters and repository fixture paths. Real model
adapters are not registered yet; the smoke test exercises evaluation and CLI
reporting with synthetic predictions.

## Current status

The current development tree implements the v0.8 multi-model reference-comparison
workflow on top of the v0.7 normalized output contract. 
Only `DummyAdapter` is registered. Gates, report writing, real model integrations,
temporal tracking, and 3D comparison remain unimplemented.

The first configured model is the reference; all remaining models are candidates.
Models execute sequentially, reusing one reference run and preserving candidate
config order. The default `common` schema mode compares exact shared keypoint
names and records excluded names; `strict` requires identical name tuples and
order. Geometric person matching supports multiple people, including reversed
pose tuple order, without relying on person IDs.

Quality measures agreement with that reference, not ground-truth accuracy.
Normalized 2D keypoint distance is `sqrt(dx^2 + dy^2) / sqrt(2)`.
Coverage is reference-successful images divided by all reference attempts.
Candidate inference and validation failures remain runtime failures and missed
reference poses. Matched-only deviation must be assessed alongside misses,
extras, failure rates, coverage, and schema exclusions.

v0.8 comparison operates in normalized 2D image coordinates. A future 3D mode
will require a common 3D coordinate-frame and unit contract before Euclidean
distance is meaningful.

Implemented so far:

- Python package using the `src/` layout
- minimal CLI entry point
- YAML-based configuration loading
- strict config schema with defaults
- config validation at path, parse, schema, and app levels
- config-specific exception hierarchy
- local quality gates via pre-commit
- pre-push test execution
- GitHub Actions CI workflow
- Ruff linting and formatting
- pytest test suite
- adapter interface via `PoseAdapter`
- normalized adapter input/output types
- deterministic built-in `DummyAdapter`
- deterministic file discovery and iteration via `FileDataSource`
- config-driven runner construction and execution
- configurable adapter warmup and prediction failure policy
- monotonic per-prediction and total runner timing capture
- deployment-oriented minimum, mean, P50, P95, P99, and maximum latency metrics
- prediction success, failure, and error-rate metrics
- sequential multi-model CLI summaries with runtime metrics, schema exclusions, and aggregate quality
- declared keypoint schemas with exact name and ordering validation
- normalized coordinate, missing-keypoint, confidence, and person-ID validation
- validation of every warmup output and measured prediction
- separate adapter execution and output validation failure counts in the CLI
- library API for optimal multi-person matching with configurable shared-point and distance requirements

Validation failures count toward error rate and are excluded from latency
samples. Inference timing is captured before validation. Adapters are
responsible for normalizing their native outputs; the validator checks the
resulting contract.

Start with the [architecture overview](docs/architecture.md) for the pipeline and layer boundaries.
Config documentation is in [docs/config.md](docs/config.md).
Adapter documentation is in [docs/adapters.md](docs/adapters.md).
The normalized output contract is in [docs/output-contract.md](docs/output-contract.md).
Data source documentation is in [docs/data.md](docs/data.md).
Runner and timing documentation is in [docs/runner.md](docs/runner.md).
Metric semantics are documented in [docs/metrics.md](docs/metrics.md).
Reference comparison, person assignment, coverage, and missing-keypoint semantics
are documented in [docs/comparison.md](docs/comparison.md).

Runnable examples and editable templates are indexed in
[docs/examples/README.md](docs/examples/README.md).

## Quick start

Clone the repository and install development dependencies:

```bash
uv sync --dev
```

Run the CLI:

```bash
uv run python -m pose_deploy_gate --version
uv run python -m pose_deploy_gate --config docs/examples/config.minimal.yaml --list-inputs
uv run python -m pose_deploy_gate --config docs/examples/config.comparison-smoke.yaml -v
uv run python -m pose_deploy_gate --input .
```

Run tests:

```bash
uv run pytest -q
```

Run local quality checks:

```bash
pre-commit run --all-files
```

## Development workflow

### Pre-commit hooks

Pre-commit hooks run automatically before committing:

- Ruff lint (`ruff --fix`)
- Ruff format (`ruff format`)

### Pre-push hook

The pre-push hook runs the pytest test suite.

Install hooks locally:

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

## Continuous integration (CI)

GitHub Actions runs on:

- pushes to `main` and feature branches with name starting with `feat/`
- pull requests

CI currently verifies:

- pre-commit checks on all files
- pytest test suite

## Roadmap

Implemented through the `v0.8` development milestone:

- adapter interface and deterministic dummy adapter
- deterministic data source iteration
- runner with warmup and timing capture
- configurable prediction failure handling
- config-driven CLI execution
- deployment-oriented latency and prediction error-rate metrics
- normalized output contract and runner validation
- classified adapter execution and output validation failures
- sequential evaluation of one reference and N candidates, reusing one reference run
- common and strict schema resolution with explicit keypoint exclusions
- multi-person matching independent of pose tuple order
- reference-relative quality metrics, unmatched poses, and comparison coverage
- candidate failures retained in runtime errors and missed reference poses
- compact and verbose CLI summaries for every model
- end-to-end synthetic multi-model regression tests

Planned next steps:

- v0.8 release preparation, version update, and release tagging
- gate evaluation through `GateEvaluator` in v0.9
- JSON and Markdown reports through `ReportWriter` in v0.9
- CI artifacts and additional CI checks
- real model adapter integration

`GateEvaluator` and `ReportWriter` will consume the central `EvaluationResult`;
gate decisions and report serialization remain outside the result object.
See the [roadmap](docs/roadmap.md) for later milestones.

## License

Apache License
