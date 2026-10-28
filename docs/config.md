# Config Reference

PoseDeployGate uses YAML to configure one reference and N candidates, shared
input discovery, runner policies, and reference-relative comparison settings.
Output paths and gate settings are reserved configuration; reports and gates
are not implemented in the current comparison pipeline.

## Loading

Configs are loaded with `load_config(config_path)`.

- Supported file extensions are `.yaml` and `.yml`.
- Files must contain a top-level YAML mapping.
- Unknown keys are rejected at every schema level.
- Path values are parsed as `pathlib.Path` values.
- `data.input_dir` must exist and must be a directory when the config is loaded.
- Relative paths are interpreted from the process working directory, not the
  config file's directory. Run repository examples from the repository root.

## CLI Usage

Run the configured pipeline from the command line:

```bash
uv run python -m pose_deploy_gate --config ./path/to/config.yaml
```

Print the deterministic input list discovered from that config:

```bash
uv run python -m pose_deploy_gate --config ./path/to/config.yaml --list-inputs
```

Add `-v` (also `--verbose` or `-verbose`) for each model's adapter type, warmup
iterations and time, latency sample count, minimum/mean/P50/P95/P99/maximum
latency, detailed reliability counts, and total runner time:

```bash
uv run python -m pose_deploy_gate --config ./path/to/config.yaml -v
```

The configuration schema supports one or more named models. The CLI passes the
loaded config to [`EvaluationSession`](evaluation.md), which treats the first
entry as the reference and runs the remaining entries as candidates sequentially
in configured order. A single-model config prints only a reference summary.
See [config.comparison.yaml](examples/config.comparison.yaml) for a multi-model
example using the built-in dummy adapter.

## Minimal Example

Start from the minimal example when you only need the default run, output, and
gate settings:

[config.minimal.yaml](examples/config.minimal.yaml)

```yaml
version: 1

data:
  input_dir: "."

models:
  - id: dummy
    adapter:
      type: "dummy"
```

## Full Example

The full example sets every currently supported section and field:

[config.full.yaml](examples/config.full.yaml)

## Schema

### `version`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `version` | Yes | literal `1` | none | Config schema version. Only `1` is currently supported. |

### `run`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `run.name` | No | string | `default-run` | Must not be empty or whitespace. |
| `run.seed` | No | integer or null | `null` | Must be non-negative when set. Parsed metadata; the session does not seed adapters or random generators. |
| `run.device` | No | `cpu`, `cuda`, or `auto` | `cpu` | Parsed preference; the current session does not apply device placement. |

### `data`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `data.input_dir` | Yes | path | none | Must exist and must be a directory. |
| `data.file_pattern` | No | string | `*` | Must not be empty or whitespace. |
| `data.recursive` | No | boolean | `false` | Enables recursive input discovery. |

Data source iteration is deterministic:

- discovery filters to files only
- ordering is based on the relative POSIX path from `data.input_dir`
- emitted `image_id` values are relative paths without the file suffix

### `models`

`models` is required and must contain at least one model. Python parses the list
as `tuple[ModelConfig, ...]`, preserving configured order. IDs must be non-blank
and unique by exact string equality. Values are never trimmed or lowercased:
`model-a` and `Model-A` are distinct IDs. A single model is valid for profiling.
The first model is the reference; the remaining models are candidates. There is
no separate role field. The reference runs exactly once, followed by each
candidate sequentially in config order. Every candidate is compared to that
same reference result, never to another candidate. The same runner policy and
data-source instance apply to all models.
The legacy top-level `adapter:` syntax is rejected.

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `models[].id` | Yes | string | none | Unique, non-blank model ID. |
| `models[].adapter` | Yes | mapping | none | Adapter configuration for this model. |
| `models[].adapter.type` | Yes | literal `dummy` | none | Only the dummy adapter is currently supported. |
| `models[].adapter.params` | No | mapping | `{}` | Adapter-specific parameters. |

### `runner`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `runner.warmup_iterations` | No | non-negative integer | `3` | Adapter calls made with the first input before measured predictions. Use `0` to disable warmup. |
| `runner.continue_on_error` | No | boolean | `false` | Records prediction failures and continues when enabled; otherwise the first prediction failure stops the run. |

Warmup, timing boundaries, and failure behavior are documented in
[docs/runner.md](runner.md).

### `comparison`

This optional section defines reference comparison settings. Omitting it uses
the defaults below. The CLI evaluation session uses these settings for schema
resolution and pose matching. See [pose matching](comparison.md)
for the assignment objective, threshold behavior, and missing-point semantics.

| Field | Type | Default | Bounds / permitted values |
| --- | --- | --- | --- |
| `comparison.schema_mode` | string | `common` | `strict` or `common` |
| `comparison.coordinate_space` | string | `image_2d_normalized` | Only `image_2d_normalized` |
| `comparison.distance_metric` | string | `normalized_euclidean_2d` | Only `normalized_euclidean_2d` |
| `comparison.min_common_keypoints` | integer | `3` | At least `1` |
| `comparison.min_pose_shared_keypoints` | integer | `3` | At least `1` |
| `comparison.min_pose_shared_fraction` | float | `0.5` | Greater than `0`, at most `1` |
| `comparison.max_pose_match_distance` | float | `0.15` | From `0` to `1`, inclusive |

`common` compares the exact-name intersection in reference order and requires
at least `min_common_keypoints` names. `strict` requires identical name tuples,
including order, regardless of schema labels; it does not apply that minimum.
Neither mode renames aliases or infers anatomical equivalence. Each output must
still conform to its own declared schema. `SchemaResolution` records compared,
reference-only, and candidate-only names; the CLI prints compared counts and
exclusion counts for both sides. Excluded names are not missing-keypoint events.

Schema compatibility and person matching use separate minima. For example,
`min_common_keypoints: 1` allows a one-name intersection in common mode, but
the default `min_pose_shared_keypoints: 3` will still prevent pose matches.
The shared fraction divides available coordinate pairs by the resolved compared
name count, not by either model's original schema size.

Distances use `sqrt(dx^2 + dy^2) / sqrt(2)` in the normalized image unit square.
The pose matching threshold applies to the mean shared-keypoint distance and is
inclusive. These measurements describe reference agreement, not ground-truth
accuracy. See [comparison semantics](comparison.md) for assignment and scoring.

v0.8 comparison operates in normalized 2D image coordinates. A future 3D mode
will require a common 3D coordinate-frame and unit contract before Euclidean
distance is meaningful. No other coordinate-space or distance-metric values
are accepted yet.

### `output`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `output.dir` | No | path | `./artifacts` | Reserved output directory path. Displayed by the CLI; no directory or report is written. |

### `gates`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `gates.enabled` | No | boolean | `false` | Reserved for deployment gate evaluation. The gate engine is not implemented yet. |

## Validation Failures

Config loading raises config-specific exceptions:

- `ConfigFileNotFoundError` when the config file path does not exist.
- `ConfigParseError` when YAML parsing fails, the file is empty, or the top
  level is not a mapping.
- `ConfigValidationError` when the file extension, schema, unknown keys, or
  application validation rules fail.

Common invalid configs include missing required fields, invalid `run.device`
values, unknown keys, empty files, malformed YAML, missing `data.input_dir`,
empty `run.name`, empty `data.file_pattern`, and negative `run.seed`.

Schema incompatibility, changing schemas, missing successful outputs, and
inconsistent image-ID sets are evaluation-time errors, not config-load checks.
With `continue_on_error: true`, candidate failures remain runtime failures and
count as unmatched reference poses on reference-successful images. Reference
failures reduce `comparison_coverage`, defined as reference-successful images
divided by all reference attempts. Missing or unexpected image IDs stop the
evaluation. See [failure and coverage rules](comparison.md#failures-and-comparison-coverage).

## Example Files

- Minimal example: [config.minimal.yaml](examples/config.minimal.yaml)
- Multiple models: [config.multiple-models.yaml](examples/config.multiple-models.yaml)
- Common-mode comparison: [config.comparison.yaml](examples/config.comparison.yaml)
- Runnable two-model smoke test: [config.comparison-smoke.yaml](examples/config.comparison-smoke.yaml)
- Runnable strict-mode comparison: [config.comparison-strict.yaml](examples/config.comparison-strict.yaml)
- Runner example: [config.runner.yaml](examples/config.runner.yaml)
- Full example: [config.full.yaml](examples/config.full.yaml)
- Recursive JPEG example: [config.recursive.yaml](examples/config.recursive.yaml)
- JPEG-only example: [config.jpg-only.yaml](examples/config.jpg-only.yaml)

Data source behavior is documented in [docs/data.md](data.md).
The [example index](examples/README.md) distinguishes runnable smoke configs
from templates whose input paths need editing.
