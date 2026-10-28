"""Command-line interface for PoseDeployGate."""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_deploy_gate import __version__
from pose_deploy_gate.adapters import AdapterError
from pose_deploy_gate.config import load_config
from pose_deploy_gate.config.exceptions import ConfigError
from pose_deploy_gate.data import DataSourceError
from pose_deploy_gate.evaluation import (
    CandidateEvaluationResult,
    EvaluationSession,
    ModelEvaluationResult,
)
from pose_deploy_gate.runner import PredictionFailureKind, RunnerError, ns_to_ms


def _format_latency_ns(value: float | int | None) -> str:
    """Format a latency in nanoseconds for CLI output."""
    if value is None:
        return "n/a"
    return f"{ns_to_ms(value):.3f} ms"


def _format_rate(value: float | None) -> str:
    """Format a ratio as a percentage for CLI output."""
    if value is None:
        return "n/a"
    return f"{100 * value:.2f}%"


def _print_model_summary(
    label: str,
    model: ModelEvaluationResult,
    *,
    verbose: bool = False,
    adapter_type: str | None = None,
) -> None:
    """Print runtime metrics already computed by the evaluation session."""
    print(f"{label}: {model.model_id}")
    print(f"  Images: {model.metrics.errors.total_attempts}")
    print(f"  P95 latency: {_format_latency_ns(model.metrics.latency.p95_ns)}")
    print(f"  Failure rate: {_format_rate(model.metrics.errors.error_rate)}")
    if verbose:
        print(f"  Adapter: {adapter_type}")
        print(f"  Warmup iterations: {model.run_result.warmup.iterations}")
        print(f"  Warmup time: {_format_latency_ns(model.run_result.warmup.total_time_ns)}")
        print()
        print("  Latency:")
        print(f"    Samples: {model.metrics.latency.sample_count}")
        for name, value in (
            ("Min", model.metrics.latency.min_ns),
            ("Mean", model.metrics.latency.mean_ns),
            ("P50", model.metrics.latency.p50_ns),
            ("P95", model.metrics.latency.p95_ns),
            ("P99", model.metrics.latency.p99_ns),
            ("Max", model.metrics.latency.max_ns),
        ):
            print(f"    {name}: {_format_latency_ns(value)}")
        print()
        print("  Reliability:")
    indent = "    " if verbose else "  "
    print(f"{indent}Successful predictions: {model.metrics.errors.successful_predictions}")
    print(f"{indent}Failed predictions: {model.metrics.errors.failed_predictions}")
    for kind, name in (
        (PredictionFailureKind.ADAPTER_EXECUTION, "Adapter execution failures"),
        (PredictionFailureKind.OUTPUT_VALIDATION, "Output validation failures"),
    ):
        count = sum(prediction.failure_kind is kind for prediction in model.run_result.predictions)
        print(f"{indent}  {name}: {count}")
    if verbose:
        print(f"    Error rate: {_format_rate(model.metrics.errors.error_rate)}")
        print()
        print(f"  Total runner time: {_format_latency_ns(model.run_result.total_time_ns)}")


def _print_candidate_summary(
    candidate: CandidateEvaluationResult,
    *,
    verbose: bool = False,
    adapter_type: str | None = None,
) -> None:
    """Print schema scope and aggregate quality without per-keypoint details."""
    _print_model_summary("Candidate", candidate.model, verbose=verbose, adapter_type=adapter_type)
    schema = candidate.schema_resolution
    compared_count = len(schema.compared_keypoints)
    print()
    print("  Schema comparison:")
    print(f"    Reference: {schema.reference_schema.name}")
    print(f"    Candidate: {schema.candidate_schema.name}")
    print(
        f"    Compared keypoints: {compared_count}/{len(schema.reference_schema.keypoint_names)} "
        f"reference, {compared_count}/{len(schema.candidate_schema.keypoint_names)} candidate"
    )
    print(f"    Reference-only keypoints excluded: {len(schema.reference_only_keypoints)}")
    print(f"    Candidate-only keypoints excluded: {len(schema.candidate_only_keypoints)}")
    quality = candidate.quality_metrics
    print()
    print("  Quality:")
    print(f"    Comparison coverage: {_format_rate(quality.comparison_coverage)}")
    print(f"    Matched poses: {quality.matched_pose_count}")
    print(
        f"    Missed reference poses: {quality.unmatched_reference_pose_count} "
        f"({_format_rate(quality.reference_pose_miss_rate)})"
    )
    print(f"    Extra candidate poses: {quality.unmatched_candidate_pose_count}")
    print(f"    Mean keypoint deviation: {_format_rate(quality.mean_keypoint_deviation)}")
    print(f"    P95 keypoint deviation: {_format_rate(quality.p95_keypoint_deviation)}")
    print(f"    Missing keypoint rate: {_format_rate(quality.candidate_missing_keypoint_rate)}")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="pose-deploy-gate",
        description="A tool for deploying pose estimation models.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the version number and exit.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the YAML config file to run.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the input file or directory to validate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no --input is provided.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        "-verbose",
        action="store_true",
        help="Show detailed runtime statistics for every model with --config.",
    )
    parser.add_argument(
        "--list-inputs",
        action="store_true",
        help="List discovered input files when used with --config.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute the CLI logic based on the parsed arguments and return an exit code."""
    if args.config is not None:
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            print(f"ERROR: {exc}")
            return 2

        print("PoseDeployGate config validation successful.")
        print(f"Config path: {args.config.resolve()}")
        print(f"Run name: {config.run.name}")
        print(f"Input directory: {config.data.input_dir.resolve()}")
        print(f"Output directory: {config.output.dir}")
        print(f"Gates enabled: {config.gates.enabled}")

        try:
            result = EvaluationSession(config).run()
        except (AdapterError, DataSourceError, RunnerError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 2

        print("PoseDeployGate evaluation completed.")
        print()
        verbose = getattr(args, "verbose", False)
        adapter_types = {model.id: model.adapter.type for model in config.models}
        _print_model_summary(
            "Reference",
            result.reference,
            verbose=verbose,
            adapter_type=adapter_types[result.reference.model_id],
        )
        for candidate in result.candidates:
            print()
            _print_candidate_summary(
                candidate, verbose=verbose, adapter_type=adapter_types[candidate.model.model_id]
            )

        if getattr(args, "list_inputs", False):
            print("Input files:")
            for index, prediction in enumerate(result.reference.run_result.predictions, start=1):
                relative_path = prediction.image.path.relative_to(config.data.input_dir).as_posix()
                print(f"  {index:03d}: {relative_path}")

        return 0

    if args.input is None:
        if args.strict:
            print("ERROR: --input is required when --strict is set.")
            return 2

        print("PoseDeployGate CLI is wired correctly.")
        print("Warning: No --input provided. Skipping validation.")
        return 0

    if not args.input.exists():
        print(f"ERROR: The specified input path '{args.input}' does not exist.")
        return 1

    path_type = "directory" if args.input.is_dir() else "file"
    print("PoseDeployGate input validation successful.")
    print(f"Resolved path: {args.input.resolve()}, Path type: ({path_type})")
    return 0


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    return run(args)
