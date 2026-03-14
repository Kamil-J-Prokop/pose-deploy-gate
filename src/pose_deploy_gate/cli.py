"""Command-line interface for PoseDeployGate."""

from __future__ import annotations

import argparse
from pathlib import Path

from pose_deploy_gate import __version__


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
        "--input",
        type=Path,
        help="Path to the input file or directory to validate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no --input is provided.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute the CLI logic based on the parsed arguments and return an exit code."""
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
