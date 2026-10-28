"""Multi-model evaluation results for downstream gate and report consumers.

These containers hold already computed results; they do not run evaluation,
recompute metrics, or validate consistency between the supplied components.
"""

from dataclasses import dataclass
from typing import Literal

from pose_deploy_gate.comparison.aggregate import QualityMetrics
from pose_deploy_gate.comparison.result import ImageComparisonResult
from pose_deploy_gate.comparison.schema import SchemaResolution
from pose_deploy_gate.metrics.result import MetricsResult
from pose_deploy_gate.runner.result import RunResult


@dataclass(frozen=True)
class ModelEvaluationResult:
    """One model's identity, evaluation role, run, and execution metrics."""

    model_id: str
    role: Literal["reference", "candidate"]
    run_result: RunResult
    metrics: MetricsResult


@dataclass(frozen=True)
class CandidateEvaluationResult:
    """A candidate's execution and quality relative to the evaluation reference.

    Unlike comparison.CandidateComparisonResult, this includes model execution
    and aggregate quality metrics. Image comparisons retain their supplied order.
    """

    model: ModelEvaluationResult
    schema_resolution: SchemaResolution
    image_comparisons: tuple[ImageComparisonResult, ...]
    quality_metrics: QualityMetrics


@dataclass(frozen=True)
class EvaluationResult:
    """One reference and its candidate comparisons, in supplied candidate order.

    Empty candidate tuples represent reference-only evaluations. Consumers share
    this reference for every candidate; gates and report serialization belong
    outside this result hierarchy.
    """

    reference: ModelEvaluationResult
    candidates: tuple[CandidateEvaluationResult, ...]
