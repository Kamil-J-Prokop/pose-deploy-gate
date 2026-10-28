# Evaluation sessions

`EvaluationSession` is the library API for sequential multi-model evaluation:

```python
from pathlib import Path

from pose_deploy_gate.config import load_config
from pose_deploy_gate.evaluation import EvaluationSession

config = load_config(Path("config.yaml"))
result = EvaluationSession(config).run()
```

The first entry in `config.models` is the reference. Each call to `run()` creates
one data-source instance and fresh model adapters and runners using the configured
runner policies. It runs the reference exactly once and computes its runtime
metrics. Then, for each remaining model in config order, it runs the candidate,
computes its runtime metrics, resolves schemas, compares outputs, and aggregates
quality before starting the next model. A single-model config returns an empty
candidate tuple. Repeated calls perform new evaluations without caching runs.

All runners receive the same data-source object. Each runner discovers inputs
when it starts, so the input directory should remain stable throughout the
session. Image IDs must be unique within each run. Every candidate must have
exactly the same image-ID set as the reference, including failed predictions.
Duplicate, missing, or unexpected IDs raise `EvaluationConsistencyError` (a
`ValueError` subclass), identifying the model and offending IDs. A prediction
marked successful without an output raises the same error. Alignment never
uses positional `zip()` and never silently drops an unmatched ID. These checks
detect changes to the ID set; they do not verify file content stability.

Schema resolution requires at least one successful output from every model,
including the reference. Successful outputs must have the same `KeypointSchema`
throughout that model's run. Missing or changing schemas raise `ValueError`;
incompatible model schemas propagate the schema resolver's `ValueError`.
The session does not guess schemas from failed predictions or adapter attributes.

Predictions join by `prediction.image.image_id`, preserving reference prediction
order regardless of candidate tuple order:

| Reference | Candidate | Quality result for this image |
| --- | --- | --- |
| Success | Success | Normal pose matching and keypoint comparison. |
| Success | Failure | One image comparison with every reference pose unmatched, no matches, and no candidate poses. |
| Failure | Success or failure | Quality unavailable; no image comparison. |

Candidate failures remain unchanged in `RunResult` with `output=None`, their
error and failure kind, and their contribution to runtime error metrics. The
session constructs the unmatched-pose comparison directly rather than inventing
a successful empty adapter output. A successful reference output with no poses
still contributes an image comparison, including when the candidate fails.

```text
compared_image_count = number of reference-successful images
comparison_coverage = reference-successful images / total reference attempts
                    = len(image_comparisons) / len(reference.run_result.predictions)
```

Thus coverage is the same across candidates and measures availability of
reference evidence. Candidate crashes cannot reduce its numerator or disappear
from the reference pose-miss denominator. They increase the missed-pose count
by the number of poses in that reference output. Reference failures reduce
coverage and exclude that image's candidate poses from quality scoring.

Deviation statistics and keypoint missing rates remain conditional on matched
poses; a crash adds no synthetic deviation or keypoint sample. A matched-only
mean can improve when a difficult image crashes, so consumers must assess it
alongside pose-miss rates, runtime failures, and coverage. For example, if the
reference succeeds on two of four images and the candidate fails on one of
those two, coverage is `2/4`, and that failed image's reference poses are misses.
Runtime metrics still include every prediction's success or failure.

Runner fail-fast and warmup errors propagate and stop the session. Schema and
alignment errors also stop it; later candidates do not run and no partial
`EvaluationResult` is returned. With `continue_on_error=True`, individual
prediction failures are retained, subject to the successful-output requirement
for schema resolution. A model with no successful outputs still raises
`ValueError`; this commit preserves the mandatory schema result contract rather
than returning quality with an unknown schema. Empty runs likewise cannot
resolve a schema, so sessions do not return zero-attempt coverage. The standalone
quality aggregator continues to report `None` for a zero denominator.

The result is an `EvaluationResult` containing a reference
`ModelEvaluationResult` and an ordered tuple of `CandidateEvaluationResult`
values. Each candidate includes its own run and runtime metrics, schema
resolution, per-image comparisons, and aggregate quality metrics. Gate evaluation,
report serialization, and CLI integration are outside this component.
