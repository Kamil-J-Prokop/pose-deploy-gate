"""Sequential orchestration of reference and candidate model evaluation."""

from dataclasses import dataclass
from typing import Literal

from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput
from pose_deploy_gate.comparison import (
    CandidateComparisonResult,
    ImageComparisonResult,
    QualityAggregator,
    QualityComparator,
    resolve_comparison_schema,
)
from pose_deploy_gate.config.models import AppConfig, ModelConfig
from pose_deploy_gate.data import FileDataSource, create_data_source
from pose_deploy_gate.evaluation.exceptions import EvaluationConsistencyError
from pose_deploy_gate.evaluation.result import (
    CandidateEvaluationResult,
    EvaluationResult,
    ModelEvaluationResult,
)
from pose_deploy_gate.metrics import MetricsEngine
from pose_deploy_gate.runner import PredictionResult, create_model_runner


@dataclass(frozen=True)
class EvaluationSession:
    """Evaluate the first configured model as reference, then candidates in order.

    Each call to ``run`` creates fresh adapters and one shared data source, and
    runs each model once. Runner errors and schema errors stop the session; no
    partial evaluation is returned. Warmup remains the runner's responsibility.
    """

    config: AppConfig

    def run(self) -> EvaluationResult:
        """Compare reference successes by image ID, counting candidate failures as misses."""
        data_source = create_data_source(self.config.data)
        reference = self._run_model(self.config.models[0], "reference", data_source)
        reference_predictions = _index_predictions(reference)
        reference_schema, reference_outputs = _resolve_model_outputs(
            reference, reference_predictions
        )
        comparator = QualityComparator()
        aggregator = QualityAggregator()
        candidates = []
        for model_config in self.config.models[1:]:
            model = self._run_model(model_config, "candidate", data_source)
            candidate_predictions = _index_predictions(model)
            missing = reference_predictions.keys() - candidate_predictions.keys()
            unexpected = candidate_predictions.keys() - reference_predictions.keys()
            if missing or unexpected:
                raise EvaluationConsistencyError(
                    f"Model {model.model_id!r} image IDs differ from reference {reference.model_id!r}; "
                    f"missing image IDs: {sorted(missing)!r}; "
                    f"unexpected image IDs: {sorted(unexpected)!r}"
                )
            candidate_schema, candidate_outputs = _resolve_model_outputs(
                model, candidate_predictions
            )
            schema = resolve_comparison_schema(
                reference_schema,
                candidate_schema,
                self.config.comparison.schema_mode,
                self.config.comparison.min_common_keypoints,
            )
            images = []
            for image_id, reference_output in reference_outputs.items():
                candidate_output = candidate_outputs.get(image_id)
                if candidate_output is None:
                    # This ID exists in the candidate run, but its prediction failed.
                    # Keep the failure in the run; do not fabricate a successful output.
                    image = ImageComparisonResult(
                        image_id=image_id,
                        pose_matches=(),
                        unmatched_reference_pose_indices=tuple(range(len(reference_output.poses))),
                        unmatched_candidate_pose_indices=(),
                    )
                else:
                    image = comparator.compare(
                        image_id,
                        reference_output,
                        candidate_output,
                        schema,
                        self.config.comparison,
                    )
                images.append(image)
            image_comparisons = tuple(images)
            comparison = CandidateComparisonResult(
                reference.model_id, model.model_id, schema, image_comparisons
            )
            # Every reference success contributes one image, even on candidate failure.
            quality = aggregator.aggregate(comparison, total_image_count=len(reference_predictions))
            candidates.append(CandidateEvaluationResult(model, schema, image_comparisons, quality))

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


def _index_predictions(model: ModelEvaluationResult) -> dict[str, PredictionResult]:
    """Index all attempts, including failures, without silently dropping duplicates."""
    predictions = {}
    for prediction in model.run_result.predictions:
        image_id = prediction.image.image_id
        if image_id in predictions:
            raise EvaluationConsistencyError(
                f"Model {model.model_id!r} has duplicate image ID {image_id!r}"
            )
        if prediction.error is None and prediction.output is None:
            raise EvaluationConsistencyError(
                f"Model {model.model_id!r} has no output for successful image {image_id!r}"
            )
        predictions[image_id] = prediction
    return predictions


def _resolve_model_outputs(
    model: ModelEvaluationResult,
    predictions: dict[str, PredictionResult],
) -> tuple[KeypointSchema, dict[str, AdapterOutput]]:
    """Resolve a stable schema and retain successful outputs in prediction order."""
    schema = None
    outputs = {}
    for image_id, prediction in predictions.items():
        if prediction.error is not None:
            continue
        output = prediction.output
        assert output is not None  # Checked by _index_predictions.
        if schema is None:
            schema = output.schema
        elif schema != output.schema:
            raise ValueError(f"Model {model.model_id!r} changes schema across successful outputs")
        outputs[image_id] = output

    if schema is None:
        raise ValueError(f"Model {model.model_id!r} has no successful output to resolve its schema")
    return schema, outputs
