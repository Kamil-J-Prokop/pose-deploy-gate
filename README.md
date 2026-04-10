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

The initial project skeleton is in place and includes:

- Python package using the `src/` layout
- minimal CLI entry point
- local quality gates via pre-commit
- pre-push test execution
- GitHub Actions CI workflow
- Ruff linting and formatting
- pytest test suite

The current CLI validates basic input paths and package wiring.

Config documentation is in [docs/config.md](docs/config.md).

## Quick start

Clone the repository and install development dependencies:

```bash
uv sync --dev
```

Run the CLI:

```bash
uv run python -m pose_deploy_gate --version
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

- pushes to `main`
- pull requests

CI currently verifies:

- pre-commit checks on all files
- pytest test suite

## Roadmap

Planned next steps:

- structured model adapter interface
- metric definitions for deployment-oriented comparison
- reproducible evaluation protocol
- reporting layer for model selection decisions

## License

Apache License
