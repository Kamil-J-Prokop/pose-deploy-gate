"""Sequential orchestration of reference and candidate model evaluation."""

from dataclasses import dataclass
from typing import Literal

from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput
from pose_deploy_gate.comparison import (
    CandidateComparisonResult,
    QualityAggregator,
    QualityComparator,
    resolve_comparison_schema,
)
from pose_deploy_gate.config.models import AppConfig, ModelConfig
from pose_deploy_gate.data import FileDataSource, create_data_source
from pose_deploy_gate.evaluation.result import (
    CandidateEvaluationResult,
    EvaluationResult,
    ModelEvaluationResult,
)
from pose_deploy_gate.metrics import MetricsEngine
from pose_deploy_gate.runner import create_model_runner


@dataclass(frozen=True)
class EvaluationSession:
    """Evaluate the first configured model as reference, then candidates in order.

    Each call to ``run`` creates fresh adapters and one shared data source, and
    runs each model once. Runner errors and schema errors stop the session; no
    partial evaluation is returned. Warmup remains the runner's responsibility.
    """

    config: AppConfig

    def run(self) -> EvaluationResult:
        """Run models sequentially and compare successful outputs by image ID."""
        data_source = create_data_source(self.config.data)
        reference = self._run_model(self.config.models[0], "reference", data_source)
        reference_schema, reference_outputs = _resolve_model_outputs(reference)
        comparator = QualityComparator()
        aggregator = QualityAggregator()
        candidates = []
        for model_config in self.config.models[1:]:
            model = self._run_model(model_config, "candidate", data_source)
            candidate_schema, candidate_outputs = _resolve_model_outputs(model)
            schema = resolve_comparison_schema(
                reference_schema,
                candidate_schema,
                self.config.comparison.schema_mode,
                self.config.comparison.min_common_keypoints,
            )
            images = tuple(
                comparator.compare(
                    image_id,
                    reference_output,
                    candidate_outputs[image_id],
                    schema,
                    self.config.comparison,
                )
                for image_id, reference_output in reference_outputs.items()
                if image_id in candidate_outputs
            )
            comparison = CandidateComparisonResult(
                reference.model_id, model.model_id, schema, images
            )
            quality = aggregator.aggregate(
                comparison, total_image_count=len(reference.run_result.predictions)
            )
            candidates.append(CandidateEvaluationResult(model, schema, images, quality))

        return EvaluationResult(reference, tuple(candidates))

    def _run_model(
        self,
        model: ModelConfig,
        role: Literal["reference", "candidate"],
        data_source: FileDataSource,
    ) -> ModelEvaluationResult:
        runner = create_model_runner(
            model,
            data_config=self.config.data,
            runner_config=self.config.runner,
            data_source=data_source,
        )
        run = runner.run()
        return ModelEvaluationResult(model.id, role, run, MetricsEngine().compute(run))


def _resolve_model_outputs(
    model: ModelEvaluationResult,
) -> tuple[KeypointSchema, dict[str, AdapterOutput]]:
    """Require unique image IDs and a stable schema across successful outputs.

    Dict insertion order preserves reference prediction order for comparisons.
    Failed predictions still contribute to runtime metrics and coverage totals.
    """
    schema = None
    outputs = {}
    seen_ids = set()
    for prediction in model.run_result.predictions:
        image_id = prediction.image.image_id
        if image_id in seen_ids:
            raise ValueError(f"Model {model.model_id!r} has duplicate image ID {image_id!r}")
        seen_ids.add(image_id)
        if prediction.error is not None:
            continue
        output = prediction.output
        if output is None:
            raise ValueError(
                f"Model {model.model_id!r} has no output for successful image {image_id!r}"
            )
        if schema is None:
            schema = output.schema
        elif schema != output.schema:
            raise ValueError(f"Model {model.model_id!r} changes schema across successful outputs")
        outputs[image_id] = output

    if schema is None:
        raise ValueError(f"Model {model.model_id!r} has no successful output to resolve its schema")
    return schema, outputs
