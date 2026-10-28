# Normalized Adapter Output Contract

The normalized output contract introduced in v0.7 remains the per-image adapter
contract for v0.8 comparison. It lets the runner and downstream consumers use
the same coordinate, schema, confidence, and person rules across runtimes.

## Coordinate System

Keypoint coordinates are normalized to the original input image:

- `x` and `y` are finite numbers in the inclusive range `[0, 1]`.
- The origin `(0, 0)` is the top-left corner.
- Positive `x` points right; positive `y` points down.
- `(1, 1)` is the bottom-right boundary; `(0.5, 0.5)` is the image center.

Coordinates refer to the original image, rather than a model's resized input,
padded tensor, or person crop. Adapters must undo any preprocessing transforms
before returning normalized coordinates. Native pixel conventions and transform
inversion belong to the adapter.

Out-of-range coordinates, NaN, and positive or negative infinity are invalid.

## Missing Keypoints And Visibility

A missing keypoint remains in its declared schema position with its original
name. Represent it with both `x=None` and `y=None`; do not remove it from the
pose or replace its coordinates with zero.

Coordinates must either both be present or both be absent. A partially missing
pair, such as `x=None, y=0.5`, is invalid.

`visible` is optional. A missing coordinate pair can have `visible=None` or
`visible=False`, but cannot have `visible=True`. A keypoint with coordinates
may have `visible=False`, for example when its position is estimated despite
occlusion. Visibility does not replace the coordinate rules.

## Declared Keypoint Schema

Every `AdapterOutput` declares a `KeypointSchema` with a non-blank `name` and a
non-empty tuple of unique, non-blank `keypoint_names`. Each pose must contain
exactly those names in exactly that order:

```python
tuple(keypoint.name for keypoint in pose.keypoints) == output.schema.keypoint_names
```

Missing entries, extra entries, duplicate names, wrong names, and reordered
entries are invalid. Missing coordinates still require an entry for that name.
Exact ordering supports deterministic serialization and comparisons, and lets
downstream code use schema positions consistently.

For a schema `("nose", "left_shoulder", "right_shoulder")`:

| Actual keypoint names | Valid? |
| --- | --- |
| `("nose", "left_shoulder", "right_shoulder")` | Yes, including when an entry has missing coordinates. |
| `("nose", "right_shoulder", "left_shoulder")` | No: wrong order, even though the set matches. |
| `("nose", "left_shoulder")` | No: missing entry. |
| `("nose", "left_shoulder", "left_shoulder")` | No: duplicate and mismatched names. |

The built-in schemas are `DUMMY_5_SCHEMA` and `COCO_17_SCHEMA`. Adapters can
declare custom schemas; there is no required universal body-keypoint taxonomy.
This validation rule concerns each output's own schema. It does not require
reference and candidate schemas to be identical: default `common` comparison
resolves shared names, while `strict` requires identical name tuples and order.

## Confidence

Both `PosePrediction.confidence` and `Keypoint.confidence` are optional. `None`
means no confidence value is supplied. Every supplied value must be finite and
within the inclusive range `[0, 1]`; NaN and infinity are invalid.

Adapters must convert native scores into this range when needed. Validation
does not calibrate confidence scores or establish that different models' scores
have the same statistical meaning.

## Person Semantics

`AdapterOutput.poses` contains zero or more poses. An empty tuple is valid when
no people are detected. Both output validation and v0.8 comparison support
multiple people; neither requires exactly one person.

`PosePrediction.person_id` is optional:

- `None` is valid, including for multiple poses in the same output.
- A non-empty string is valid.
- An empty string (`""`) is invalid.
- Non-`None` IDs must be unique within one image's `AdapterOutput`.

IDs can repeat across images or inference calls. The same person can therefore
keep the same ID across a sequence. Validation does not enforce global ID
uniqueness or perform tracking. IDs are compared exactly as supplied; the
validator does not trim or normalize them.

For reference comparison, people are matched geometrically within each image,
independently of pose tuple order and supplied person IDs. IDs only support an
advisory agreement flag in the standalone matching result. There is no temporal
tracking or cross-frame identity matching.

## Adapter Responsibility

Adapters must convert native runtime output into this contract. That includes
mapping native keypoint names and order, restoring the original image coordinate
frame, normalizing coordinates and confidence, and representing missing points.

The PDG validator verifies conformance of the supplied values and structure. It
does not perform model-specific normalization, rename or reorder keypoints,
repair coordinates, or convert native scores. A coordinate in `[0, 1]` can still
refer to the wrong image frame; only the adapter has enough information to
ensure the original-image requirement is satisfied.

## Example And Public Validation API

This output contains one person and a missing left shoulder. The missing entry
remains in the schema order:

```python
from pose_deploy_gate.adapters.schema import KeypointSchema
from pose_deploy_gate.adapters.types import AdapterOutput, Keypoint, PosePrediction
from pose_deploy_gate.validation import (
    AdapterOutputValidationError,
    AdapterOutputValidator,
    OutputValidationCode,
    OutputValidationIssue,
)

schema = KeypointSchema(
    name="shoulders_3",
    keypoint_names=("nose", "left_shoulder", "right_shoulder"),
)
output = AdapterOutput(
    schema=schema,
    poses=(
        PosePrediction(
            person_id="person-1",
            confidence=0.9,
            keypoints=(
                Keypoint(name="nose", x=0.5, y=0.2, confidence=0.95),
                Keypoint(name="left_shoulder", x=None, y=None, visible=False),
                Keypoint(name="right_shoulder", x=0.6, y=0.4, confidence=0.8),
            ),
        ),
    ),
)

validator = AdapterOutputValidator()
assert validator.validate(output) == ()
validator.validate_or_raise(output)
```

`validate(output)` returns a tuple of all detected `OutputValidationIssue`
objects, or `()` for valid output. Each issue has a `path`, an
`OutputValidationCode`, and an explanatory `message`. Validation aggregates
schema, identity, coordinate, visibility, and confidence issues.

`validate_or_raise(output)` returns `None` for valid output and raises
`AdapterOutputValidationError` otherwise. The exception's `issues` tuple
preserves all detected issues. Constructing the output dataclasses alone does
not run output validation.

## Runner, Metrics, And CLI Behavior

The runner validates every warmup output and every measured prediction. Any
invalid warmup output aborts the run, regardless of `continue_on_error`.

For measured predictions, inference timing is captured immediately after
`adapter.predict()` and before validation, so validation does not add to the
prediction's inference latency.

With `continue_on_error=False` (the default), invalid output raises
`RunnerExecutionError` with the image ID and validation details. With
`continue_on_error=True`, the runner records a failed `PredictionResult` with
`output=None`, an error message, and
`failure_kind=PredictionFailureKind.OUTPUT_VALIDATION`, then processes the next
image. Adapter execution failures use `ADAPTER_EXECUTION`.

Validation failures count toward failure totals and error rate. They contribute
no latency samples. Completed CLI runs show counts for each failure kind
without dumping individual issues:

```text
  Successful predictions: 98
  Failed predictions: 2
    Adapter execution failures: 0
    Output validation failures: 2
```

Fail-fast CLI errors include the image ID, issue path, code, and explanation,
and return exit code `2`.

## v0.8 Reference Comparison

`EvaluationSession` uses the first configured model as its reference and all
remaining models as candidates. Models run sequentially, with the reference
run reused for every candidate. Predictions align by image ID, not tuple
position; missing, extra, or duplicate IDs are consistency errors.

`common` mode compares the intersection of exact keypoint names in reference
order, requiring at least `min_common_keypoints` (default `3`). Names unique to
one model are recorded as exclusions and do not affect matching or keypoint
rates. `strict` mode requires identical names and ordering; schema labels may
differ. Adapters must supply consistent anatomical meanings for shared names;
the resolver does not translate aliases or infer meaning from point counts.

Within each image, matching maximizes eligible person correspondences and then
minimizes the sum of accepted pair costs. Each cost is the mean shared-keypoint
distance, where a keypoint distance is
`sqrt((reference.x - candidate.x)^2 + (reference.y - candidate.y)^2) / sqrt(2)`.
Unmatched reference and candidate poses remain explicit. Reversed person output
order can therefore have zero deviation when the coordinates agree.

On a reference-successful image, a candidate inference or validation failure
remains a runtime failure and leaves all reference poses unmatched. Reference
failures make quality unavailable for those images. Coverage is
`reference-successful images / total reference attempts`. Candidate failures
cannot disappear from pose-miss statistics by being omitted. Deviation and
missing-keypoint rates remain conditional on matched poses; read them with
misses and runtime failures. Each model must have a stable schema and at least
one successful output, or evaluation stops without a partial result.

Comparison measures reference-relative agreement, not ground-truth accuracy.
The reference's mistakes are not independently detected. Full matching,
exclusion, failure, and metric semantics are documented in
[multi-model reference comparison](comparison.md).

## 3D Boundary

v0.8 comparison operates in normalized 2D image coordinates. A future 3D mode
will require a common 3D coordinate-frame and unit contract before Euclidean
distance is meaningful. Depth, 3D coordinates, physical units, and camera/world
frame conversion are not part of the current adapter or comparison contract.

See also the [adapter guide](adapters.md), [runner guide](runner.md), and
[metric semantics](metrics.md).
