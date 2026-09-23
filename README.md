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

## Current status

The `v0.6.0` development milestone adds deployment-oriented latency and
prediction-error metrics derived from config-driven runner results.

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
- CLI run summaries with latency and reliability metrics

This milestone provides a stable metrics boundary for upcoming validation,
gate, and reporting work.

Config documentation is in [docs/config.md](docs/config.md).
Adapter documentation is in [docs/adapters.md](docs/adapters.md).
Data source documentation is in [docs/data.md](docs/data.md).
Runner and timing documentation is in [docs/runner.md](docs/runner.md).
Metric semantics are documented in [docs/metrics.md](docs/metrics.md).

Example configs are in [docs/examples](docs/examples).

## Quick start

Clone the repository and install development dependencies:

```bash
uv sync --dev
```

Run the CLI:

```bash
uv run python -m pose_deploy_gate --version
uv run python -m pose_deploy_gate --config docs/examples/config.minimal.yaml
uv run python -m pose_deploy_gate --config docs/examples/config.runner.yaml
uv run python -m pose_deploy_gate --config docs/examples/config.minimal.yaml --list-inputs
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

Implemented through the `v0.6.0` development milestone:

- adapter interface
- deterministic dummy adapter
- deterministic data source iteration
- runner with warmup and timing capture
- configurable prediction failure handling
- config-driven CLI execution
- deployment-oriented latency metrics
- prediction error-rate metrics
- CLI latency and reliability summary

Planned next steps:

- output validation and gate evaluation
- report generation and CI artifacts

## License

Apache License
