# Pose matching

`PoseMatcher` finds one-to-one person correspondences between a reference and a
candidate output for the same image. It maximizes the number of valid matches,
then minimizes their total matching distance. Matching is independent for each
image; it does not track identities across frames.

The matcher is a library API. Comparison orchestration, final keypoint scoring,
aggregation, and CLI reporting are not connected yet. `DummyAdapter` remains
the only registered adapter.

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

The caller supplies validated outputs in the original image coordinate frame
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
also holds final `KeypointComparison` values for the later scoring layer.
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

Tests use the existing dummy adapter and directly constructed normalized pose
objects and cost matrices. They cover empty and unequal counts, crossing poses,
greedy failures, missing points, threshold boundaries, advisory IDs, and index
partitioning. Assignment count and total distance are independently checked
against exhaustive enumeration of small partial matchings. Additional real or
simulated data sources are outside this commit's scope.
