# Config Reference

PoseDeployGate uses a YAML config file to describe an evaluation run: where
input data comes from, which named models to configure, where outputs should be written,
and whether deployment gates are enabled.

## Loading

Configs are loaded with `load_config(config_path)`.

- Supported file extensions are `.yaml` and `.yml`.
- Files must contain a top-level YAML mapping.
- Unknown keys are rejected at every schema level.
- Path values are parsed as `pathlib.Path` values.
- `data.input_dir` must exist and must be a directory when the config is loaded.

## CLI Usage

Run the configured pipeline from the command line:

```bash
uv run python -m pose_deploy_gate --config ./path/to/config.yaml
```

Print the deterministic input list discovered from that config:

```bash
uv run python -m pose_deploy_gate --config ./path/to/config.yaml --list-inputs
```

The configuration schema supports one or more named models. The CLI and runner
currently execute configs containing exactly one model. Configs containing
multiple models load successfully through `load_config`, but CLI execution returns
a clear error with exit code `2` until multi-model execution is implemented.

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
| `run.seed` | No | integer or null | `null` | When set, must be non-negative. |
| `run.device` | No | `cpu`, `cuda`, or `auto` | `cpu` | Selects the runtime device preference. |

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
Models have no roles or comparison semantics at this stage.
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
the defaults below. The library's `PoseMatcher` consumes these settings; CLI
comparison execution is not connected yet. See [pose matching](comparison.md)
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

Coordinate space and distance metric are explicit so future 3D support can add
new values without silently changing the meaning of existing configurations.

### `output`

| Field | Required | Type | Default | Notes |
| --- | --- | --- | --- | --- |
| `output.dir` | No | path | `./artifacts` | Output directory path. It is parsed but not required to exist during config loading. |

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

## Example Files

- Minimal example: [config.minimal.yaml](examples/config.minimal.yaml)
- Multiple models: [config.multiple-models.yaml](examples/config.multiple-models.yaml)
- Runner example: [config.runner.yaml](examples/config.runner.yaml)
- Full example: [config.full.yaml](examples/config.full.yaml)
- Recursive JPEG example: [config.recursive.yaml](examples/config.recursive.yaml)
- JPEG-only example: [config.jpg-only.yaml](examples/config.jpg-only.yaml)

Data source behavior is documented in [docs/data.md](data.md).
