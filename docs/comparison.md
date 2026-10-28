# Multi-model reference comparison

v0.8 comparison evaluates one reference model against zero or more candidates.
The first entry in `models` is the reference; every remaining entry is a
candidate. `EvaluationSession` runs the reference exactly once, then runs and
compares each candidate sequentially in configured order. Candidates are never
compared to each other. A single-model config produces runtime metrics for the
reference and an empty candidate tuple.

These metrics describe agreement with the reference, not ground-truth accuracy.
The reference can be wrong, and identical incorrect outputs have zero deviation.
No annotations, accuracy labels, PCK, or OKS calculation are used. Interpret
deviation together with pose misses, extra poses, runtime failures, and coverage.

Run the two-model example from the repository root:

```bash
uv run python -m pose_deploy_gate --config docs/examples/config.comparison-smoke.yaml -v
```

This uses synthetic `DummyAdapter` predictions and fixture paths. The dummy
adapter does not decode the files or perform real model inference. See
[configuration](config.md) for settings and [evaluation sessions](evaluation.md)
for the result hierarchy, execution policy, and CLI output.

## Schema resolution

Every output must first pass validation against its own declared schema: its
keypoint entries must have the exact names and order declared by that output.
Pairwise comparison is a separate step and supports two modes:

| Mode | Compatibility rule | Compared names and exclusions |
| --- | --- | --- |
| `common` (default) | At least `min_common_keypoints` exact names are shared, default `3`. Different schema names, lengths, and ordering are allowed. | Intersection in reference order. Names unique to either schema are recorded as exclusions in that schema's order. |
| `strict` | `reference.keypoint_names == candidate.keypoint_names`: identical names and order. Schema labels may differ. | All names are compared; there are no exclusions. `min_common_keypoints` is not a strict-mode minimum. |

Names are compared exactly, without case folding, aliases, or inferred taxonomy
translation. A shared name must represent the same anatomical point in both
adapters; the resolver cannot verify that meaning. Coordinate and unit agreement
are also adapter responsibilities.

For example, these schemas share three names despite differing order:

```text
reference: (nose, left_wrist, right_wrist, left_ankle)
candidate: (right_wrist, nose, left_wrist, custom_tip)

compared_keypoints:       (nose, left_wrist, right_wrist)
reference_only_keypoints: (left_ankle,)
candidate_only_keypoints: (custom_tip,)
```

`common` mode accepts this pair at the default minimum; `strict` rejects it.
`SchemaResolution` preserves both original schemas and all three name tuples.
The CLI shows schema names, compared counts against both schemas, and exclusion
counts. Exact excluded names are available in the result object. Excluded names
do not count as missing keypoints and do not enter distance or missing-rate
denominators; a small shared subset therefore limits what agreement can show.

Schema resolution happens after each model's run. Each model must supply at
least one successful output and the same `KeypointSchema` across its successful
outputs. An incompatible pair, changing schema, or entirely failed model stops
the evaluation; no partial result is returned.

## Image alignment

Predictions are joined by `prediction.image.image_id`, never by positional
`zip()`. Each candidate must have the same complete image-ID set as the
reference, including failed attempts. Duplicate, missing, or unexpected IDs
raise `EvaluationConsistencyError`. Image comparisons preserve reference
prediction order, independently of candidate prediction tuple order.

All models receive the same file data-source instance, but each runner discovers
files when it starts. Keep input paths and contents stable throughout the run;
ID-set checks cannot detect changed image contents.

## Multi-person matching

`PoseMatcher` finds one-to-one person correspondences between a reference and a
candidate output for the same image. It maximizes the number of valid matches,
then minimizes their total matching distance. Matching is independent for each
image; it does not track identities across frames.

The matcher, per-image quality comparator, and quality aggregator are library
APIs used by `EvaluationSession` and the config-driven CLI. The CLI prints
aggregate quality and schema exclusions for every candidate; per-keypoint
details remain available in the result objects. `DummyAdapter` remains the
only registered adapter.

## Public API

```python
from pose_deploy_gate.comparison import PoseMatcher, resolve_comparison_schema
from pose_deploy_gate.config.models import ComparisonConfig

# reference and candidate are validated AdapterOutput values for the same image.
config = ComparisonConfig()
schema = resolve_comparison_schema(
    reference.schema,
    candidate.schema,
    config.schema_mode,
    config.min_common_keypoints,
)
result = PoseMatcher().match(reference, candidate, schema, config)
```

The caller supplies validated outputs in the normalized original-image coordinate frame
and a compatible resolved schema. The matcher does not load images, run
adapters, repair outputs, or resolve schemas itself. Inputs are not mutated.

The exported frozen dataclasses are:

| Type / field | Meaning |
| --- | --- |
| `PoseAssignment.reference_pose_index` | Index in the original reference output's `poses` tuple. |
| `PoseAssignment.candidate_pose_index` | Index in the original candidate output's `poses` tuple. |
| `PoseAssignment.matching_cost` | Original mean shared-keypoint distance, without any solver penalty. |
| `PoseMatchingResult.pose_matches` | Tuple of accepted `PoseAssignment` values, sorted by reference index. |
| `PoseMatchingResult.unmatched_reference_pose_indices` | Ascending tuple of all remaining reference indices. |
| `PoseMatchingResult.unmatched_candidate_pose_indices` | Ascending tuple of all remaining candidate indices. |
| `PoseMatchingResult.person_id_matching` | Advisory ID agreement across accepted matches, defined below. |

Each input person appears exactly once, either in a match or in that side's
unmatched indices. Empty inputs are valid, including one empty side. No
assignment solver is called if either side is empty.

`PoseAssignment` describes correspondence only. The separate `PoseMatch` type
also holds final `KeypointComparison` values for the quality comparison layer.
The matching cost is not a final pose quality score.

## Shared keypoints and missing coordinates

Matching costs come from `compute_pose_match_cost()`. For each possible person
pair, it looks up corresponding keypoints by resolved name. A keypoint is
shared only when both people supply both coordinates. Its distance is:

```text
sqrt((reference.x - candidate.x)^2 + (reference.y - candidate.y)^2) / sqrt(2)
```

Coordinates lie in `[0, 1]`, so each distance and their mean lie in `[0, 1]`.
The mean uses only shared points. A pair must satisfy both:

```text
shared_count >= min_pose_shared_keypoints
shared_count / len(schema.compared_keypoints) >= min_pose_shared_fraction
```

The defaults are three shared points and a shared fraction of `0.5`. With five
resolved points, either one or two missing points still leaves enough evidence
to match. With only three resolved points, losing even one fails the default
minimum count. Both requirements can be configured. A pair with no shared
points is always ineligible.

In `common` schema mode, `compared_keypoints` is the intersection of exact names
from both schemas. Keypoints belonging only to one schema do not enter the cost
or shared-fraction denominator; `SchemaResolution` records those exclusions.
Missing coordinates within the intersection do count against the fraction.
Each pose still follows its own declared schema, representing unavailable
points with `None` coordinates rather than removing schema entries.

Pose confidence, keypoint confidence, and visibility flags do not affect
matching. An occluded point with supplied coordinates still participates;
adapters represent unavailable coordinates as `None`. Missing points do not
contribute a zero distance. The later scoring layer can record presence and an
unavailable deviation separately.

The algorithm assigns whole people. Keypoint names supply evidence for each
possible person pairing; keypoints are never assigned independently across
different people.

The normalized distance is measured in the unit square, not in pixels or world
units. Dividing by `sqrt(2)` maps opposite corners to distance `1`. Image width
and height are normalized separately, so this is not pixel Euclidean distance
divided by the original image's diagonal. CLI deviation percentages are this
distance multiplied by 100, not an accuracy percentage.

## Optimal assignment and final rejection

A valid pair meets the shared-point requirements and has original matching
cost **less than or equal to** `max_pose_match_distance` (default `0.15`). The
boundary is inclusive, with no added floating-point tolerance.

The matcher builds a reference-by-candidate cost matrix and uses
[`scipy.optimize.linear_sum_assignment()`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
Rectangular matrices are supported, including five reference poses and three
candidate poses. To express the objective, every ineligible or above-threshold
pair receives the same finite sentinel cost:

```text
k = min(reference_pose_count, candidate_pose_count)
sentinel = k + 1
```

The solver selects exactly `k` pairs. Their valid distances sum to at most `k`,
so eliminating one sentinel always improves the total more than any possible
change in valid distances. This maximizes valid-match count first. For the same
valid-match count, minimizing the solver total minimizes the sum of original
valid distances. The complete rectangular matrix provides filler assignments;
dummy rows and columns are unnecessary.

**Final rejection occurs after assignment.** The matcher checks each selected
pair's original cost and discards pairs that are ineligible or exceed the
threshold. Rejected indices are reported as unmatched on both sides. Solver
sentinels are never returned as matching costs.

The threshold must also influence the solver matrix. Minimizing raw costs and
only then applying rejection would not guarantee maximum valid-match count.
For example, at threshold `0.15`:

| | Candidate A | Candidate B |
| --- | ---: | ---: |
| Reference A | 0.00 | 0.14 |
| Reference B | 0.14 | 0.16 |

Raw minimization chooses the diagonal and rejects `0.16`, leaving one match.
The implemented objective chooses both `0.14` pairs, retaining two valid matches.

Equally optimal assignments have no promised identity tie-breaker across SciPy
versions. Output collections are always ordered by original indices, but an
ambiguous pair of overlapping or identical poses can have multiple equally
valid correspondences. Person IDs do not break these ties.

For separated people, reversing the candidate tuple preserves geometric
correspondence. If reference indices `0` and `1` are people A on the left and B
on the right, a candidate tuple `(B, A)` yields `(reference=0, candidate=1)` and
`(reference=1, candidate=0)`. Identical coordinates give zero keypoint deviation.
Neither tuple position nor `person_id` determines the pairing.

## Unmatched poses

An unmatched reference pose has no eligible accepted candidate correspondence.
It can mean a candidate missed that person, supplied too few shared coordinates,
was beyond the distance threshold, or lost a competing assignment. A candidate
inference or validation failure leaves every reference pose in that image
unmatched. An unmatched candidate pose has no accepted reference correspondence.

These are reference-relative observations, not ground-truth false negatives or
false positives. An entirely wrong reference can make a correct candidate appear
extra. A distance-rejected pair can leave both a reference miss and a candidate
extra. All indices refer to the original per-image pose tuples, including after
reordering; they are not persistent person identities.

## Advisory person-ID agreement

`person_id_matching` is `True` only when at least one accepted match exists and
every accepted pair has two supplied, exactly equal `person_id` strings. It is
`False` when a matched pair has different or missing IDs, or no pairs match.
Two missing IDs do not count as agreement. Unmatched people do not participate.

This flag never changes costs, assignments, eligibility, or success/failure.
It reports observed agreement for accepted pairs in this image, not proof of a
shared ID policy or temporal identity. Existing adapter-output validation still
applies to empty IDs and duplicate supplied IDs within one output.

## Validation scope

Unit tests cover empty and unequal counts, crossing poses, greedy failures,
missing points, threshold boundaries, advisory IDs, and index partitioning.
Assignment count and total distance are independently checked against exhaustive
enumeration of small partial matchings. Config-driven regression tests use
synthetic adapters with real runners, validation, schema resolution, matching,
aggregation, and CLI formatting. They include one or three candidates, shared
schemas, strict rejection, shifted and reordered people, missed/extra people,
missing keypoints, and inference/validation failures. These tests establish
pipeline behavior, not real-model accuracy or performance.

## Per-image quality comparison

```python
from pose_deploy_gate.comparison import QualityComparator

comparison = QualityComparator().compare(
    image_id, reference, candidate, schema, config
)
```

`compare()` accepts validated outputs for the same image and a compatible resolved
schema, just like the matcher. It runs `PoseMatcher` and returns an
`ImageComparisonResult`, preserving original pose indices, matching costs, and
unmatched indices. Each match contains one `KeypointComparison` per resolved
name, in reference schema order. Inputs are not mutated.

When both points have coordinates, deviation uses the normalized image distance
defined above. Reference-present/candidate-missing records a missing event as
`reference_present=True`, `candidate_present=False`, and `deviation=None`.
Whenever the reference is missing, deviation is `None` regardless of candidate
presence: the reference supplies no coordinate for comparison.
These records are not zero-distance samples. Confidence and visibility do not
change coordinate presence. Schema-excluded keypoints are not evaluated.

## Aggregate quality metrics

```python
from pose_deploy_gate.comparison import QualityAggregator

metrics = QualityAggregator().aggregate(comparison, total_image_count=100)
```

Aggregation combines the `ImageComparisonResult` values in one
`CandidateComparisonResult`. `compared_image_count` is the number of supplied
image comparisons. `comparison_coverage` divides that count by
`total_image_count`, which can include images that failed before comparison. If
the total is omitted, the supplied comparisons are treated as the complete set.
Rates with a zero denominator and coverage for an empty set are `None`.

The session supplies a comparison for every reference-successful image, even
on candidate failure. Its coverage formula and unavailable-image rules are
specified under [failures and comparison coverage](#failures-and-comparison-coverage).

Pose rates use each model's own pose count:

```text
reference_pose_miss_rate = unmatched_reference / (matched + unmatched_reference)
candidate_extra_pose_rate = unmatched_candidate / (matched + unmatched_candidate)
```

Keypoint deviation statistics include only samples where both reference and
candidate coordinates are present. Mean, P50, P95, and maximum are `None` when
there are no such samples. Percentiles reuse the metrics module's NumPy
`method="inverted_cdf"` strategy, so they always select an observed deviation.

The candidate missing-keypoint rate counts only matched-pose keypoints for which
the reference is present. A candidate point missing against a present reference
enters the numerator; a missing reference enters neither the numerator nor the
denominator. Per-keypoint metrics use the same rules and follow resolved
reference-schema order.

## Failures and comparison coverage

With `runner.continue_on_error: true`, an adapter execution failure or invalid
output remains a failed prediction with `output=None`, an error message, and
its failure kind. It contributes to runtime error rate and not to successful
latency samples. The session applies these rules after image-ID alignment:

| Reference attempt | Candidate attempt | Comparison result |
| --- | --- | --- |
| Success | Success | Normal comparison, including successful empty pose outputs. |
| Success | Failure | One image comparison with all reference poses unmatched, no matches, and no candidate poses. |
| Failure | Either | Quality unavailable for this image; no image comparison. |

In a completed session, every reference success produces exactly one image
comparison per candidate:

```text
compared_image_count = reference-successful images
comparison_coverage = reference-successful images / total reference attempts
                    = len(image_comparisons) / len(reference.run_result.predictions)
```

Coverage measures reference evidence availability and is the same for every
candidate. A successful reference image with zero poses still counts. Candidate
failures cannot disappear by reducing coverage or the reference pose-miss
denominator; their reference poses are counted as misses. Reference failures
reduce coverage and exclude the image's candidate poses from quality scoring.
For example, if a reference succeeds on two of four images and a candidate
fails on one of those two, coverage is `50%`, and that image's reference poses
are all misses.

Deviation and missing-keypoint rates describe matched poses only. No synthetic
distance penalty is added for a crash or unmatched person. A conditional mean
can improve when a difficult image crashes, so it must be read alongside pose
misses, runtime failures, and coverage. A candidate failure on an empty reference
image adds no pose misses but remains a runtime failure.

Warmup errors and fail-fast prediction errors stop the evaluation. Entirely
failed models cannot resolve their schema and also stop it. The session never
returns a completed zero-attempt evaluation; standalone aggregation returns
`None` for rates with a zero denominator. See [failure policy](evaluation.md).

## Temporal and 3D boundaries

Every image is matched independently. v0.8 performs no temporal tracking,
cross-frame association, or persistent identity inference. Repeated person IDs
do not introduce those capabilities.

v0.8 comparison operates in normalized 2D image coordinates. A future 3D mode
will require a common 3D coordinate-frame and unit contract before Euclidean
distance is meaningful. No 3D coordinates, depth alignment, camera/world
transforms, or 3D comparison mode are implemented. The supported config values
are `image_2d_normalized` and `normalized_euclidean_2d` only.
