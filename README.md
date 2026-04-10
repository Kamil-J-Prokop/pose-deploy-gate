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

Version `0.2.0` establishes the configuration foundation for the project.

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

This milestone makes configuration a first-class subsystem and provides the base for the upcoming adapter, runner, and metrics work.

Config documentation is in [docs/config.md](docs/config.md).

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

Planned next steps:

- adapter interface and dummy adapter
- deterministic data source iteration
- runner with warmup and timing capture
- deployment-oriented metrics
- output validation and gate evaluation
- report generation and CI artifacts

## License

Apache License
