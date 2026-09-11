"""Command-line interface for PoseDeployGate."""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_deploy_gate import __version__
from pose_deploy_gate.adapters import AdapterError
from pose_deploy_gate.config import load_config
from pose_deploy_gate.config.exceptions import ConfigError
from pose_deploy_gate.data import DataSourceError
from pose_deploy_gate.runner import RunnerError, create_runner, ns_to_ms


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
        print(f"Adapter: {config.adapter.type}")
        print(f"Output directory: {config.output.dir}")
        print(f"Gates enabled: {config.gates.enabled}")

        try:
            runner = create_runner(config)
            result = runner.run()

        except AdapterError as exc:
            print(f"ERROR: {exc}")
            return 2
        except DataSourceError as exc:
            print(f"ERROR: {exc}")
            return 2
        except RunnerError as exc:
            print(f"ERROR: {exc}")
            return 2

        print("PoseDeployGate run completed.")
        print(f"Input files: {len(result.predictions)}")
        print(f"Warmup iterations: {result.warmup.iterations}")
        print(f"Successful predictions: {result.successful_predictions}")
        print(f"Failed predictions: {result.failed_predictions}")
        print(f"Average inference time: {ns_to_ms(result.average_inference_time_ns):.3f} ms")
        print(
            f"Total measured inference time: {ns_to_ms(result.measured_inference_time_ns):.3f} ms"
        )
        print(f"Total runner time: {ns_to_ms(result.total_time_ns):.3f} ms")

        if getattr(args, "list_inputs", False):
            print("Input files:")
            for index, prediction in enumerate(result.predictions, start=1):
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
