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
session. Image IDs must be unique within each run; duplicates raise `ValueError`
instead of silently replacing predictions during alignment.

Schema resolution requires at least one successful output from every model,
including the reference. Successful outputs must have the same `KeypointSchema`
throughout that model's run. Missing or changing schemas raise `ValueError`;
incompatible model schemas propagate the schema resolver's `ValueError`.
The session does not guess schemas from failed predictions or adapter attributes.

Comparisons join successful outputs by `prediction.image.image_id`, preserving
reference prediction order. Failed, missing, and candidate-only images do not
produce comparisons. Coverage uses **all reference predictions** as its
denominator, including failures. If both models have successful outputs but no
shared successful image IDs, the candidate has empty comparisons and zero
coverage. Runtime metrics still include every prediction's success or failure.

Runner fail-fast and warmup errors propagate and stop the session. Schema and
alignment errors also stop it; later candidates do not run and no partial
`EvaluationResult` is returned. With `continue_on_error=True`, individual
prediction failures are retained, subject to the successful-output requirement
for schema resolution.

The result is an `EvaluationResult` containing a reference
`ModelEvaluationResult` and an ordered tuple of `CandidateEvaluationResult`
values. Each candidate includes its own run and runtime metrics, schema
resolution, per-image comparisons, and aggregate quality metrics. Gate evaluation,
report serialization, and CLI integration are outside this component.
